"""
Environment Setup Script
========================
Run this ONCE before the agent. It:
1. Downloads the Waterbirds dataset
2. Fine-tunes a ViT-B/16 on it (with spurious background correlation)
3. Creates the sandboxed environment directory with train/val/test splits
   (test split is hidden from the agent)

Usage:
    CUDA_VISIBLE_DEVICES=1 python environment/setup.py --env_root /data3/ishaplan/pref_model_env
"""

import argparse
import os
import sys
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm


# ---------------------------------------------------------------------------
# 1. Download Waterbirds
# ---------------------------------------------------------------------------

WATERBIRDS_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


def download_waterbirds(raw_dir: Path):
    """Download and extract the Waterbirds dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "waterbirds.tar.gz"

    if not tar_path.exists():
        print("Downloading Waterbirds dataset...")
        subprocess.run(["wget", "-q", "-O", str(tar_path), WATERBIRDS_URL], check=True)

    extract_dir = raw_dir / "waterbird_complete95_forest2water2"
    if not extract_dir.exists():
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(raw_dir)

    return extract_dir


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class WaterbirdsDataset(Dataset):
    """Waterbirds with group labels (bird_type x background)."""

    def __init__(self, root_dir, split, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        metadata = pd.read_csv(self.root_dir / "metadata.csv")
        # split: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        self.data = metadata[metadata["split"] == split_map[split]].reset_index(drop=True)

        self.filenames = self.data["img_filename"].tolist()
        self.labels = self.data["y"].tolist()              # 0=landbird, 1=waterbird
        self.places = self.data["place"].tolist()           # 0=land, 1=water
        # group = 2*y + place => 4 groups
        self.groups = [2 * y + p for y, p in zip(self.labels, self.places)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.groups[idx]


# ---------------------------------------------------------------------------
# 3. Fine-tune ViT-B/16
# ---------------------------------------------------------------------------

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def finetune_vit(data_dir: Path, save_path: Path, epochs=10, lr=1e-4, batch_size=64):
    """Fine-tune ViT-B/16 on Waterbirds with ERM (standard training, no debiasing).
    This intentionally learns the spurious correlation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_ds = WaterbirdsDataset(data_dir, "train", get_transforms(train=True))
    val_ds = WaterbirdsDataset(data_dir, "val", get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Load pretrained ViT-B/16 from timm
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_avg_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels, groups in train_loader:
            imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validate with group breakdown
        model.eval()
        group_correct = {g: 0 for g in range(4)}
        group_total = {g: 0 for g in range(4)}
        with torch.no_grad():
            for imgs, labels, groups in val_loader:
                imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
                logits = model(imgs)
                preds = logits.argmax(1)
                for i in range(len(labels)):
                    g = groups[i]
                    group_correct[g] += (preds[i] == labels[i]).item()
                    group_total[g] += 1

        group_accs = {g: group_correct[g] / max(group_total[g], 1) for g in range(4)}
        avg_acc = sum(group_correct.values()) / sum(group_total.values())
        worst_group_acc = min(group_accs.values())

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | "
              f"Val Avg: {avg_acc:.3f} | Val Worst-Group: {worst_group_acc:.3f}")
        print(f"  Groups: {', '.join(f'G{g}: {group_accs[g]:.3f}' for g in range(4))}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "vit_base_patch16_224",
                "num_classes": 2,
            }, save_path)
            print(f"  -> Saved best model (avg_acc={avg_acc:.3f})")

    return model


# ---------------------------------------------------------------------------
# 4. Prepare the sandboxed environment
# ---------------------------------------------------------------------------

def prepare_environment(data_dir: Path, model_path: Path, env_root: Path):
    """Create the agent's sandboxed workspace."""
    # Directories the agent can see
    agent_data = env_root / "data" / "waterbirds"
    agent_model = env_root / "model"
    agent_output = env_root / "output"
    # Hidden from agent: test set for the judge
    judge_dir = env_root / ".judge"

    for d in [agent_data, agent_model, agent_output, judge_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load full metadata
    metadata = pd.read_csv(data_dir / "metadata.csv")

    # --- Agent gets train + val ---
    agent_meta = metadata[metadata["split"].isin([0, 1])].copy()
    agent_meta["split_name"] = agent_meta["split"].map({0: "train", 1: "val"})

    # Copy images for agent (train + val)
    print("Copying train/val images for agent...")
    for _, row in agent_meta.iterrows():
        src = data_dir / row["img_filename"]
        dst = agent_data / "images" / Path(row["img_filename"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    # Save agent metadata
    agent_csv = agent_meta[["img_filename", "y", "place", "split"]].copy()
    agent_csv.columns = ["filename", "label", "background", "split"]
    agent_csv["filename"] = agent_csv["filename"].apply(lambda x: Path(x).name)
    agent_csv["split"] = agent_csv["split"].map({0: "train", 1: "val"})
    agent_csv.to_csv(agent_data / "metadata.csv", index=False)

    # --- Judge gets test set (hidden) ---
    test_meta = metadata[metadata["split"] == 2].copy()
    print("Copying test images for judge (hidden from agent)...")
    for _, row in test_meta.iterrows():
        src = data_dir / row["img_filename"]
        dst = judge_dir / "images" / Path(row["img_filename"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    test_csv = test_meta[["img_filename", "y", "place"]].copy()
    test_csv.columns = ["filename", "label", "background"]
    test_csv["filename"] = test_csv["filename"].apply(lambda x: Path(x).name)
    # group = 2*label + background
    test_csv["group"] = 2 * test_csv["label"] + test_csv["background"]
    test_csv.to_csv(judge_dir / "test_metadata.csv", index=False)

    # --- Copy model ---
    shutil.copy2(model_path, agent_model / "vit_waterbirds.pt")

    # Save model config
    config = {
        "model_name": "vit_base_patch16_224",
        "num_classes": 2,
        "class_names": ["landbird", "waterbird"],
        "background_names": ["land", "water"],
        "groups": {
            "0": "landbird on land (majority)",
            "1": "landbird on water (minority)",
            "2": "waterbird on land (minority)",
            "3": "waterbird on water (majority)",
        },
        "image_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    with open(agent_model / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Copy prompt ---
    prompt_src = Path(__file__).parent.parent / "agent" / "prompt.txt"
    if prompt_src.exists():
        shutil.copy2(prompt_src, env_root / "prompt.txt")

    print(f"\nEnvironment ready at: {env_root}")
    print(f"  Agent workspace: {env_root}  (data/, model/, output/)")
    print(f"  Judge data:      {judge_dir}  (hidden from agent)")
    print(f"  Agent should work inside: {env_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True,
                        help="Root directory for the sandboxed environment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip ViT fine-tuning if model already exists")
    args = parser.parse_args()

    env_root = Path(args.env_root)
    raw_dir = env_root / ".raw"
    model_path = env_root / ".raw" / "vit_waterbirds.pt"

    # Step 1: Download
    data_dir = download_waterbirds(raw_dir)

    # Step 2: Fine-tune (or skip)
    if args.skip_training and model_path.exists():
        print(f"Skipping training, using existing model at {model_path}")
    else:
        finetune_vit(data_dir, model_path, epochs=args.epochs,
                     lr=args.lr, batch_size=args.batch_size)

    # Step 3: Prepare sandboxed environment
    prepare_environment(data_dir, model_path, env_root)


if __name__ == "__main__":
    main()
