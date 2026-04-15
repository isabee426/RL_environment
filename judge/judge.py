"""
Judge Script
============
Evaluates the agent's modified model on the held-out test set.

Scoring:
  - 0.0 if output files are missing or model can't load
  - 0.0 if >10% of parameters were changed (surgical constraint violated)
  - 0.0 if average accuracy < 70% (model is broken)
  - Otherwise: score = worst-group accuracy (continuous 0.0 - 1.0)

Usage:
    CUDA_VISIBLE_DEVICES=1 python judge/judge.py --env_root /data3/ishaplan/pref_model_env
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm


class TestDataset(Dataset):
    def __init__(self, judge_dir, transform):
        self.judge_dir = Path(judge_dir)
        self.transform = transform
        self.meta = pd.read_csv(self.judge_dir / "test_metadata.csv")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(self.judge_dir / "images" / row["filename"]).convert("RGB")
        img = self.transform(img)
        return img, int(row["label"]), int(row["group"])


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def count_param_diff(model_a, model_b):
    """Count fraction of parameters that differ between two models."""
    total = 0
    changed = 0
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b, f"Parameter mismatch: {name_a} vs {name_b}"
        n = param_a.numel()
        total += n
        changed += (param_a != param_b).sum().item()
    return changed / total if total > 0 else 0.0


def evaluate(model, dataloader, device):
    """Evaluate model, return per-group and overall accuracy."""
    model.eval()
    group_correct = {}
    group_total = {}

    with torch.no_grad():
        for imgs, labels, groups in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(1)

            for i in range(len(labels)):
                g = groups[i].item()
                if g not in group_correct:
                    group_correct[g] = 0
                    group_total[g] = 0
                group_correct[g] += (preds[i] == labels[i]).item()
                group_total[g] += 1

    group_accs = {g: group_correct[g] / group_total[g] for g in sorted(group_total)}
    avg_acc = sum(group_correct.values()) / sum(group_total.values())
    worst_group_acc = min(group_accs.values())

    return avg_acc, worst_group_acc, group_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    args = parser.parse_args()

    env_root = Path(args.env_root)
    judge_dir = env_root / ".judge"
    output_dir = env_root / "output"
    original_model_path = env_root / "model" / "vit_waterbirds.pt"
    fixed_model_path = output_dir / "model_fixed.pt"
    analysis_path = output_dir / "analysis.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "score": 0.0,
        "reason": "",
        "details": {},
    }

    # ---- Check 1: Files exist ----
    if not fixed_model_path.exists():
        results["reason"] = "FAIL: /output/model_fixed.pt not found"
        print(json.dumps(results, indent=2))
        return

    if not analysis_path.exists():
        results["reason"] = "FAIL: /output/analysis.txt not found"
        print(json.dumps(results, indent=2))
        return

    # ---- Check 2: Load models ----
    try:
        original_model = load_model(original_model_path)
        fixed_model = load_model(fixed_model_path)
    except Exception as e:
        results["reason"] = f"FAIL: Could not load model: {e}"
        print(json.dumps(results, indent=2))
        return

    # ---- Check 3: Parameter diff ----
    param_diff_frac = count_param_diff(original_model, fixed_model)
    results["details"]["param_diff_fraction"] = param_diff_frac

    if param_diff_frac > 0.10:
        results["reason"] = (
            f"FAIL: {param_diff_frac:.1%} of parameters changed (limit: 10%). "
            "Intervention was not surgical."
        )
        print(json.dumps(results, indent=2))
        return

    # ---- Check 4: Evaluate both models ----
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = TestDataset(judge_dir, test_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Evaluate original
    original_model = original_model.to(device)
    orig_avg, orig_wg, orig_groups = evaluate(original_model, test_loader, device)
    original_model = original_model.cpu()

    # Evaluate fixed
    fixed_model = fixed_model.to(device)
    fixed_avg, fixed_wg, fixed_groups = evaluate(fixed_model, test_loader, device)
    fixed_model = fixed_model.cpu()

    results["details"]["original"] = {
        "avg_accuracy": round(orig_avg, 4),
        "worst_group_accuracy": round(orig_wg, 4),
        "group_accuracies": {str(g): round(a, 4) for g, a in orig_groups.items()},
    }
    results["details"]["fixed"] = {
        "avg_accuracy": round(fixed_avg, 4),
        "worst_group_accuracy": round(fixed_wg, 4),
        "group_accuracies": {str(g): round(a, 4) for g, a in fixed_groups.items()},
    }
    results["details"]["improvement"] = {
        "avg_accuracy_delta": round(fixed_avg - orig_avg, 4),
        "worst_group_accuracy_delta": round(fixed_wg - orig_wg, 4),
    }

    # ---- Check 5: Average accuracy floor ----
    if fixed_avg < 0.70:
        results["reason"] = (
            f"FAIL: Average accuracy {fixed_avg:.1%} is below 70% floor. "
            "Model is broken."
        )
        print(json.dumps(results, indent=2))
        return

    # ---- Score ----
    results["score"] = round(fixed_wg, 4)
    results["reason"] = (
        f"PASS: Worst-group accuracy = {fixed_wg:.1%} "
        f"(original: {orig_wg:.1%}, delta: {fixed_wg - orig_wg:+.1%}). "
        f"Avg accuracy = {fixed_avg:.1%}. "
        f"Parameters changed: {param_diff_frac:.2%}."
    )

    print(json.dumps(results, indent=2))

    # Also save to file
    with open(output_dir / "judge_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
