"""
Judge: Transcoders on Vision Transformers
=========================================
Evaluates the agent's transcoder against the baseline SAE on the held-out test set.

Scoring (0.0 - 1.0):
  1. Architecture check: is it actually a skip transcoder? (pass/fail gate)
  2. Reconstruction quality: MSE on test MLP outputs (must beat or match baseline SAE)
  3. Sparsity: L0 must be in [20, 50] range (TopK=32 target)
  4. Interpretability: per-concept AUROC on test set
  5. Comparison: how many concepts does the transcoder beat the SAE on?

Final score = weighted combination of reconstruction + interpretability + comparison.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 judge/judge.py --env_root /data3/ishaplan/pref_model_env
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def load_transcoder(checkpoint_path):
    """Load the agent's transcoder and verify it has skip transcoder architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint)
    architecture = checkpoint.get("architecture", "unknown")
    input_dim = checkpoint.get("input_dim", 768)
    output_dim = checkpoint.get("output_dim", 768)
    dict_size = checkpoint.get("dict_size", None)
    topk = checkpoint.get("topk", 32)

    # Check for skip transcoder components
    has_skip = any("skip" in k.lower() for k in state_dict.keys())
    has_encoder = any("enc" in k.lower() for k in state_dict.keys())
    has_decoder = any("dec" in k.lower() for k in state_dict.keys())

    return {
        "state_dict": state_dict,
        "architecture": architecture,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "dict_size": dict_size,
        "topk": topk,
        "has_skip": has_skip,
        "has_encoder": has_encoder,
        "has_decoder": has_decoder,
        "param_keys": list(state_dict.keys()),
    }


class GenericTranscoder(nn.Module):
    """Flexible loader for the agent's transcoder."""
    def __init__(self, state_dict, input_dim, output_dim, dict_size):
        super().__init__()
        # Try to reconstruct the architecture from state dict keys
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dict_size = dict_size

        # Create layers matching the state dict
        for key, param in state_dict.items():
            name = key.replace(".", "_")
            if param.dim() >= 2:
                setattr(self, name, nn.Linear(param.shape[1], param.shape[0], bias=False))
            else:
                self.register_buffer(name, param)

        # Load the state dict
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        raise NotImplementedError("Use the agent's own forward pass")


def find_params(state_dict):
    """Flexibly find encoder/decoder/skip params regardless of naming convention.

    Handles both 'W_enc'/'b_enc' style and 'encoder.weight'/'encoder.bias' style.
    """
    enc_weight = enc_bias = dec_weight = dec_bias = skip_weight = skip_bias = None

    for key, param in state_dict.items():
        k = key.lower()
        # Encoder: W_enc, encoder.weight, enc_weight, etc.
        if ("w_enc" == k or (("enc" in k or "encoder" in k) and ("weight" in k or param.dim() == 2)))  \
           and "dec" not in k and "skip" not in k and enc_weight is None:
            enc_weight = param
        elif ("b_enc" == k or (("enc" in k or "encoder" in k) and ("bias" in k or param.dim() == 1))) \
             and "dec" not in k and "skip" not in k and enc_bias is None:
            enc_bias = param
        # Decoder: W_dec, decoder.weight, dec_weight, etc.
        elif ("w_dec" == k or (("dec" in k or "decoder" in k) and ("weight" in k or param.dim() == 2))) \
             and "enc" not in k and "skip" not in k and dec_weight is None:
            dec_weight = param
        elif ("b_dec" == k or (("dec" in k or "decoder" in k) and ("bias" in k or param.dim() == 1))) \
             and "enc" not in k and "skip" not in k and dec_bias is None:
            dec_bias = param
        # Skip: W_skip, skip.weight, etc.
        elif ("w_skip" == k or ("skip" in k and ("weight" in k or param.dim() == 2))) \
             and skip_weight is None:
            skip_weight = param
        elif ("b_skip" == k or ("skip" in k and ("bias" in k or param.dim() == 1))) \
             and skip_bias is None:
            skip_bias = param

    return enc_weight, enc_bias, dec_weight, dec_bias, skip_weight, skip_bias


def compute_feature_activations(state_dict, inputs, topk, device):
    """Run the transcoder encoder to get sparse feature activations."""
    enc_weight, enc_bias, _, _, _, _ = find_params(state_dict)

    if enc_weight is None:
        return None

    enc_weight = enc_weight.to(device)
    if enc_bias is not None:
        enc_bias = enc_bias.to(device)

    x = torch.tensor(inputs, dtype=torch.float32).to(device)
    z_pre = x @ enc_weight.T
    if enc_bias is not None:
        z_pre = z_pre + enc_bias

    # TopK
    topk_vals, topk_idx = torch.topk(z_pre, topk, dim=-1)
    z = torch.zeros_like(z_pre)
    z.scatter_(-1, topk_idx, torch.relu(topk_vals))

    return z.cpu().numpy()


def compute_reconstruction(state_dict, inputs, topk, device):
    """Reconstruct MLP outputs from MLP inputs using the transcoder."""
    enc_weight, enc_bias, dec_weight, dec_bias, skip_weight, skip_bias = find_params(state_dict)

    if enc_weight is None or dec_weight is None:
        return None

    enc_weight = enc_weight.to(device)
    if enc_bias is not None: enc_bias = enc_bias.to(device)
    dec_weight = dec_weight.to(device)
    if dec_bias is not None: dec_bias = dec_bias.to(device)
    if skip_weight is not None: skip_weight = skip_weight.to(device)
    if skip_bias is not None: skip_bias = skip_bias.to(device)

    x = torch.tensor(inputs, dtype=torch.float32).to(device)

    # Encode
    z_pre = x @ enc_weight.T
    if enc_bias is not None:
        z_pre = z_pre + enc_bias

    # TopK
    topk_vals, topk_idx = torch.topk(z_pre, topk, dim=-1)
    z = torch.zeros_like(z_pre)
    z.scatter_(-1, topk_idx, torch.relu(topk_vals))

    # Decode
    x_hat = z @ dec_weight.T
    if dec_bias is not None:
        x_hat = x_hat + dec_bias

    # Skip connection
    if skip_weight is not None:
        x_hat = x_hat + x @ skip_weight.T
        if skip_bias is not None:
            x_hat = x_hat + skip_bias

    return x_hat.cpu().numpy()


def compute_interpretability(activations, concept_labels, min_positive=10):
    """Compute per-concept best AUROC from feature activations."""
    concepts = concept_labels.columns.tolist()
    results = {}

    for concept in concepts:
        labels = concept_labels[concept].values
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        if n_pos < min_positive or n_neg < min_positive:
            continue

        best_auroc = 0.0
        best_feat = -1

        for feat_idx in range(activations.shape[1]):
            feat_acts = activations[:, feat_idx]
            if feat_acts.std() < 1e-8:
                continue
            try:
                auroc = roc_auc_score(labels, feat_acts)
                auroc = max(auroc, 1 - auroc)  # handle flipped features
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_feat = feat_idx
            except ValueError:
                continue

        if best_feat >= 0:
            results[concept] = {"feature_idx": int(best_feat), "auroc": round(best_auroc, 4)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    args = parser.parse_args()

    env_root = Path(args.env_root)
    judge_dir = env_root / ".judge"
    output_dir = env_root / "output"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "score": 0.0,
        "reason": "",
        "details": {},
    }

    # ---- Check 1: Required files ----
    transcoder_path = output_dir / "transcoder.pt"
    mapping_path = output_dir / "feature_mapping.json"
    analysis_path = output_dir / "analysis.txt"

    for path, name in [(transcoder_path, "transcoder.pt"),
                       (mapping_path, "feature_mapping.json"),
                       (analysis_path, "analysis.txt")]:
        if not path.exists():
            results["reason"] = f"FAIL: /output/{name} not found"
            print(json.dumps(results, indent=2))
            return

    # ---- Check 2: Architecture verification ----
    try:
        tc_info = load_transcoder(transcoder_path)
    except Exception as e:
        results["reason"] = f"FAIL: Could not load transcoder: {e}"
        print(json.dumps(results, indent=2))
        return

    results["details"]["architecture"] = {
        "declared": tc_info["architecture"],
        "has_skip_connection": tc_info["has_skip"],
        "has_encoder": tc_info["has_encoder"],
        "has_decoder": tc_info["has_decoder"],
        "dict_size": tc_info["dict_size"],
        "param_keys": tc_info["param_keys"],
    }

    if not tc_info["has_encoder"] or not tc_info["has_decoder"]:
        results["reason"] = "FAIL: Transcoder missing encoder or decoder components"
        print(json.dumps(results, indent=2))
        return

    if not tc_info["has_skip"]:
        results["reason"] = (
            "FAIL: No skip connection found in transcoder. "
            "A skip transcoder must have W_skip parameters."
        )
        results["score"] = 0.1  # partial credit for attempting
        print(json.dumps(results, indent=2))
        return

    if tc_info["dict_size"] is not None and tc_info["dict_size"] < 4096:
        results["reason"] = (
            f"FAIL: Dictionary size {tc_info['dict_size']} < 4096 minimum"
        )
        print(json.dumps(results, indent=2))
        return

    # ---- Check 3: Reconstruction quality on test set ----
    test_inputs = np.load(judge_dir / "mlp_inputs_test.npy")
    test_outputs = np.load(judge_dir / "mlp_outputs_test.npy")
    concept_labels_test = pd.read_csv(judge_dir / "concept_labels_test.csv")

    with open(judge_dir / "judge_meta.json") as f:
        judge_meta = json.load(f)
    baseline_mse = judge_meta["baseline_sae_mse"]

    # Reconstruct with transcoder
    topk = tc_info.get("topk", 32)
    recon = compute_reconstruction(tc_info["state_dict"], test_inputs, topk, device)

    if recon is None:
        results["reason"] = "FAIL: Could not run transcoder forward pass"
        print(json.dumps(results, indent=2))
        return

    tc_mse = float(((test_outputs - recon) ** 2).mean())
    output_var = float(test_outputs.var())
    normalized_mse = tc_mse / max(output_var, 1e-8)

    results["details"]["reconstruction"] = {
        "transcoder_mse": round(tc_mse, 6),
        "baseline_sae_mse": round(baseline_mse, 6),
        "output_variance": round(output_var, 6),
        "normalized_mse": round(normalized_mse, 6),
        "beats_baseline": tc_mse < baseline_mse,
    }

    # ---- Check 4: Sparsity ----
    tc_activations = compute_feature_activations(
        tc_info["state_dict"], test_inputs, topk, device
    )

    if tc_activations is None:
        results["reason"] = "FAIL: Could not compute transcoder feature activations"
        print(json.dumps(results, indent=2))
        return

    avg_l0 = float((tc_activations > 0).sum(axis=1).mean())
    results["details"]["sparsity"] = {
        "avg_l0": round(avg_l0, 1),
        "target_l0": topk,
        "in_range": 20 <= avg_l0 <= 50,
    }

    # ---- Check 5: Interpretability ----
    # Transcoder features
    tc_interp = compute_interpretability(tc_activations, concept_labels_test)

    # Baseline SAE features
    baseline_activations = np.load(judge_dir / "baseline_sae_activations.npy")
    sae_interp = compute_interpretability(baseline_activations, concept_labels_test)

    results["details"]["interpretability"] = {
        "transcoder_concepts_found": len(tc_interp),
        "baseline_sae_concepts_found": len(sae_interp),
    }

    # Compare per-concept
    shared_concepts = set(tc_interp.keys()) & set(sae_interp.keys())
    tc_wins = 0
    sae_wins = 0
    per_concept = {}

    for concept in shared_concepts:
        tc_auroc = tc_interp[concept]["auroc"]
        sae_auroc = sae_interp[concept]["auroc"]
        per_concept[concept] = {
            "transcoder_auroc": tc_auroc,
            "baseline_sae_auroc": sae_auroc,
            "winner": "transcoder" if tc_auroc > sae_auroc else "baseline_sae",
        }
        if tc_auroc > sae_auroc:
            tc_wins += 1
        else:
            sae_wins += 1

    results["details"]["comparison"] = {
        "shared_concepts": len(shared_concepts),
        "transcoder_wins": tc_wins,
        "baseline_sae_wins": sae_wins,
        "win_rate": round(tc_wins / max(len(shared_concepts), 1), 4),
        "per_concept": per_concept,
    }

    # ---- Compute final score ----
    # Reconstruction score (0-0.3): how good is MSE relative to baseline
    if tc_mse <= baseline_mse * 0.8:
        recon_score = 0.3  # significantly better
    elif tc_mse <= baseline_mse:
        recon_score = 0.2  # better or equal
    elif tc_mse <= baseline_mse * 1.5:
        recon_score = 0.1  # somewhat worse but reasonable
    else:
        recon_score = 0.0  # much worse

    # Sparsity score (0-0.1)
    sparsity_score = 0.1 if 20 <= avg_l0 <= 50 else 0.0

    # Interpretability score (0-0.3): average AUROC across concepts
    if tc_interp:
        avg_tc_auroc = np.mean([v["auroc"] for v in tc_interp.values()])
        interp_score = min(0.3, (avg_tc_auroc - 0.5) * 1.5)  # scale 0.5-0.7 -> 0-0.3
        interp_score = max(0.0, interp_score)
    else:
        interp_score = 0.0

    # Comparison score (0-0.3): what fraction of concepts does transcoder beat SAE
    if shared_concepts:
        win_rate = tc_wins / len(shared_concepts)
        comparison_score = win_rate * 0.3
    else:
        comparison_score = 0.0

    final_score = recon_score + sparsity_score + interp_score + comparison_score

    results["score"] = round(final_score, 4)
    results["details"]["score_breakdown"] = {
        "reconstruction": round(recon_score, 4),
        "sparsity": round(sparsity_score, 4),
        "interpretability": round(interp_score, 4),
        "comparison": round(comparison_score, 4),
    }

    results["reason"] = (
        f"Score: {final_score:.2f}/1.0 | "
        f"Recon MSE: {tc_mse:.6f} (baseline: {baseline_mse:.6f}) | "
        f"L0: {avg_l0:.0f} | "
        f"Avg AUROC: {avg_tc_auroc if tc_interp else 0:.3f} | "
        f"TC wins {tc_wins}/{len(shared_concepts)} concepts vs SAE"
    )

    print(json.dumps(results, indent=2))

    with open(output_dir / "judge_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
