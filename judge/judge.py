"""
Judge: Mechanistic Unlearning
=============================
Evaluates the agent's unlearned model on held-out test images.

Scoring:
  - Forget score (0-1): How well were target concepts erased?
    Per-concept: 1.0 if accuracy <= 55%, 0.0 if accuracy >= original, linear between
  - Retain score (0-1): How well were other concepts preserved?
    Per-concept: 1.0 if accuracy within 5% of original, 0.0 if dropped >20%, linear between
  - Localization score (0-1): Did the agent provide causal evidence for localization?
  - Final score: 0.35 * forget + 0.35 * retain + 0.15 * localization + 0.15 * has_all_outputs

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score


class TestDataset(Dataset):
    def __init__(self, img_dir, filenames, transform):
        self.img_dir = Path(img_dir)
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.filenames[idx]).convert("RGB")
        return self.transform(img), idx


def load_model(checkpoint_path, device):
    """Load a CLIPConceptClassifier from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        visual = clip_model.visual
    except ImportError:
        import timm
        visual = timm.create_model("vit_base_patch16_224", pretrained=True)

    class CLIPConceptClassifier(nn.Module):
        def __init__(self, visual, num_concepts):
            super().__init__()
            self.visual = visual
            self.head = nn.Linear(512, num_concepts)
        def forward(self, x):
            features = self.visual(x)
            if features.dim() == 3:
                features = features[:, 0, :]
            return self.head(features)

    model = CLIPConceptClassifier(visual, checkpoint["num_concepts"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def evaluate_concepts(model, dataloader, labels_df, concept_cols, device):
    """Evaluate per-concept binary accuracy."""
    all_logits = []
    with torch.no_grad():
        for imgs, indices in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    preds = (all_logits > 0).float().numpy()
    labels = labels_df[concept_cols].values.astype(float)

    per_concept_acc = {}
    for i, concept in enumerate(concept_cols):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        if y_true.sum() < 5 or (1 - y_true).sum() < 5:
            continue
        per_concept_acc[concept] = round(float(accuracy_score(y_true, y_pred)), 4)

    return per_concept_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    args = parser.parse_args()

    env_root = Path(args.env_root)
    judge_dir = env_root / ".judge"
    output_dir = env_root / "output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"score": 0.0, "reason": "", "details": {}}

    # ---- Check 1: Required files ----
    unlearned_path = output_dir / "unlearned_model.pt"
    localization_path = output_dir / "localization.json"
    analysis_path = output_dir / "analysis.txt"

    output_score = 0.0
    for path, name, weight in [
        (unlearned_path, "unlearned_model.pt", 0.6),
        (localization_path, "localization.json", 0.2),
        (analysis_path, "analysis.txt", 0.2),
    ]:
        if path.exists():
            output_score += weight
        else:
            results["details"][f"missing_{name}"] = True

    if not unlearned_path.exists():
        results["reason"] = "FAIL: /output/unlearned_model.pt not found"
        print(json.dumps(results, indent=2))
        return

    results["details"]["output_completeness"] = round(output_score, 2)

    # ---- Check 2: Load config ----
    with open(judge_dir / "judge_config.json") as f:
        config = json.load(f)

    forget_concepts = config["forget_concepts"]
    retain_concepts = config["retain_concepts"]
    concept_cols = config["concept_cols"]

    # ---- Check 3: Load test data ----
    test_labels = pd.read_csv(judge_dir / "test_labels.csv")
    test_filenames = test_labels["filename"].tolist()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])

    test_ds = TestDataset(judge_dir / "images", test_filenames, transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # ---- Check 4: Evaluate original model ----
    print("Evaluating original model...")
    original_model = load_model(judge_dir / "original_model.pt", device)
    original_accs = evaluate_concepts(original_model, test_loader, test_labels, concept_cols, device)
    original_model = original_model.cpu()

    results["details"]["original_model"] = {
        "forget_accs": {c: original_accs.get(c, 0) for c in forget_concepts},
        "retain_accs_mean": round(np.mean([original_accs.get(c, 0) for c in retain_concepts if c in original_accs]), 4),
    }

    # ---- Check 5: Evaluate unlearned model ----
    print("Evaluating unlearned model...")
    try:
        unlearned_model = load_model(unlearned_path, device)
    except Exception as e:
        results["reason"] = f"FAIL: Could not load unlearned model: {e}"
        print(json.dumps(results, indent=2))
        return

    unlearned_accs = evaluate_concepts(unlearned_model, test_loader, test_labels, concept_cols, device)
    unlearned_model = unlearned_model.cpu()

    # ---- Check 6: Forget score ----
    forget_scores = []
    forget_details = {}
    for concept in forget_concepts:
        orig = original_accs.get(concept, 0.5)
        unlearned = unlearned_accs.get(concept, 0.5)

        # Score: 1.0 if accuracy <= 55% (near chance), 0.0 if >= original
        if unlearned <= 0.55:
            score = 1.0
        elif unlearned >= orig:
            score = 0.0
        else:
            score = (orig - unlearned) / (orig - 0.55)
            score = max(0.0, min(1.0, score))

        forget_scores.append(score)
        forget_details[concept] = {
            "original_acc": orig,
            "unlearned_acc": unlearned,
            "score": round(score, 4),
        }

    avg_forget = np.mean(forget_scores) if forget_scores else 0.0
    results["details"]["forget"] = {
        "per_concept": forget_details,
        "avg_score": round(avg_forget, 4),
    }

    # ---- Check 7: Retain score ----
    retain_scores = []
    retain_details = {}
    for concept in retain_concepts:
        if concept not in original_accs or concept not in unlearned_accs:
            continue
        orig = original_accs[concept]
        unlearned = unlearned_accs[concept]
        drop = orig - unlearned

        # Score: 1.0 if drop <= 5%, 0.0 if drop >= 20%
        if drop <= 0.05:
            score = 1.0
        elif drop >= 0.20:
            score = 0.0
        else:
            score = 1.0 - (drop - 0.05) / 0.15
            score = max(0.0, min(1.0, score))

        retain_scores.append(score)
        retain_details[concept] = {
            "original_acc": orig,
            "unlearned_acc": unlearned,
            "drop": round(drop, 4),
            "score": round(score, 4),
        }

    avg_retain = np.mean(retain_scores) if retain_scores else 0.0
    results["details"]["retain"] = {
        "per_concept": retain_details,
        "avg_score": round(avg_retain, 4),
        "concepts_within_5pct": sum(1 for s in retain_scores if s == 1.0),
        "concepts_evaluated": len(retain_scores),
    }

    # ---- Check 8: Localization score ----
    localization_score = 0.0
    if localization_path.exists():
        try:
            with open(localization_path) as f:
                loc = json.load(f)

            has_method = bool(loc.get("method"))
            has_locations = bool(loc.get("forget_concept_locations"))
            has_evidence = False
            if has_locations:
                for concept, info in loc["forget_concept_locations"].items():
                    if info.get("evidence"):
                        has_evidence = True
                        break

            localization_score = 0.0
            if has_method: localization_score += 0.3
            if has_locations: localization_score += 0.3
            if has_evidence: localization_score += 0.4

            results["details"]["localization"] = {
                "has_method": has_method,
                "has_locations": has_locations,
                "has_evidence": has_evidence,
                "score": round(localization_score, 4),
            }
        except Exception:
            pass

    # ---- Final score ----
    final_score = (
        0.35 * avg_forget +
        0.35 * avg_retain +
        0.15 * localization_score +
        0.15 * output_score
    )

    results["score"] = round(final_score, 4)
    results["details"]["score_breakdown"] = {
        "forget": round(0.35 * avg_forget, 4),
        "retain": round(0.35 * avg_retain, 4),
        "localization": round(0.15 * localization_score, 4),
        "output_completeness": round(0.15 * output_score, 4),
    }

    # Summary
    forget_accs_str = ", ".join(f"{c}: {forget_details[c]['unlearned_acc']:.0%}" for c in forget_concepts if c in forget_details)
    results["reason"] = (
        f"Score: {final_score:.2f}/1.0 | "
        f"Forget: {avg_forget:.2f} ({forget_accs_str}) | "
        f"Retain: {avg_retain:.2f} ({sum(1 for s in retain_scores if s == 1.0)}/{len(retain_scores)} within 5%) | "
        f"Localization: {localization_score:.2f}"
    )

    print(json.dumps(results, indent=2))
    with open(output_dir / "judge_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
