"""
Environment Setup: Mechanistic Unlearning on CLIP ViT
=====================================================
The agent must selectively unlearn specific visual concepts from a CLIP ViT-B/16
while preserving all other capabilities.

Setup provides:
  - CLIP ViT-B/16 fine-tuned as a multi-label concept classifier on CelebA
  - Train/val images with concept labels
  - A list of target concepts to FORGET and concepts to RETAIN
  - Linear probes per concept (so the agent can verify its own work)

Judge gets:
  - Hidden test images + concept labels
  - The original (pre-unlearning) model for comparison

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 environment/setup.py --env_root /data3/ishaplan/pref_model_env
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Download CelebA
# ---------------------------------------------------------------------------

def download_celeba(raw_dir: Path):
    """Download CelebA attributes and images."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    attr_path = raw_dir / "list_attr_celeba.txt"
    img_dir = raw_dir / "img_align_celeba"

    if not attr_path.exists():
        print("Downloading CelebA attributes...")
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
            str(attr_path), quiet=False
        )

    if not img_dir.exists() or len(list(img_dir.glob("*.jpg"))) < 1000:
        print("Downloading CelebA images...")
        import gdown
        zip_path = raw_dir / "img_celeba.zip"
        if not zip_path.exists():
            gdown.download(
                "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                str(zip_path), quiet=False
            )
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(raw_dir)], check=True)

    # Parse attributes
    with open(attr_path) as f:
        n_images = int(f.readline().strip())
        attr_names = f.readline().strip().split()
        rows = []
        for line in f:
            parts = line.strip().split()
            fname = parts[0]
            if (img_dir / fname).exists():
                attrs = [1 if int(x) == 1 else 0 for x in parts[1:]]
                rows.append([fname] + attrs)

    df = pd.DataFrame(rows, columns=["filename"] + attr_names)
    return df, img_dir, attr_names


# ---------------------------------------------------------------------------
# 2. Fine-tune CLIP as concept classifier
# ---------------------------------------------------------------------------

class CelebADataset(Dataset):
    def __init__(self, img_dir, df, concept_cols, transform):
        self.img_dir = Path(img_dir)
        self.filenames = df["filename"].tolist()
        self.labels = df[concept_cols].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.filenames[idx]).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.labels[idx])


class CLIPConceptClassifier(nn.Module):
    """CLIP ViT-B/16 with a linear classification head for multi-label concept prediction."""
    def __init__(self, visual_encoder, num_concepts, hidden_dim=512):
        super().__init__()
        self.visual = visual_encoder
        # Freeze most of the model, only train last 2 blocks + head
        for param in self.visual.parameters():
            param.requires_grad = False
        # Unfreeze last 2 transformer blocks
        if hasattr(self.visual, 'transformer'):  # open_clip
            for block in self.visual.transformer.resblocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif hasattr(self.visual, 'blocks'):  # timm
            for block in self.visual.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        self.head = nn.Linear(hidden_dim, num_concepts)

    def forward(self, x):
        features = self.visual(x)
        if features.dim() == 3:  # [B, tokens, dim]
            features = features[:, 0, :]  # CLS token
        return self.head(features)


def finetune_classifier(model, train_loader, val_loader, concept_cols,
                        epochs=5, lr=1e-4, device="cpu"):
    """Fine-tune the concept classifier."""
    model = model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = (logits > 0).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | Val Acc: {val_acc:.3f}")

    return model


# ---------------------------------------------------------------------------
# 3. Choose forget/retain concepts
# ---------------------------------------------------------------------------

def choose_concepts(concept_df, concept_cols, n_forget=5):
    """Pick concepts to forget that are well-represented and interesting.

    We want:
    - Forget concepts that have enough positive examples (>5%)
    - A mix of easy and hard concepts
    - Retain concepts that are diverse
    """
    # Compute positive rates
    pos_rates = {}
    for col in concept_cols:
        pos_rates[col] = concept_df[col].mean()

    # Filter to concepts with 10-40% positive rate (well-balanced)
    balanced = {c: r for c, r in pos_rates.items() if 0.10 < r < 0.40}

    if len(balanced) < n_forget + 5:
        # Fallback: use any concepts with >5% positive rate
        balanced = {c: r for c, r in pos_rates.items() if r > 0.05}

    # Pick forget concepts — choose visually distinctive ones
    preferred_forget = [
        "Eyeglasses", "Wearing_Hat", "Bald", "Bangs", "Goatee",
        "Mustache", "Sideburns", "Gray_Hair", "Blond_Hair", "Rosy_Cheeks"
    ]
    forget_concepts = []
    for c in preferred_forget:
        if c in balanced and len(forget_concepts) < n_forget:
            forget_concepts.append(c)

    # Fill remaining from balanced concepts
    for c in sorted(balanced.keys()):
        if c not in forget_concepts and len(forget_concepts) < n_forget:
            forget_concepts.append(c)

    retain_concepts = [c for c in concept_cols if c not in forget_concepts]

    return forget_concepts, retain_concepts


# ---------------------------------------------------------------------------
# 4. Train per-concept linear probes (for agent to use)
# ---------------------------------------------------------------------------

def train_probes(model, dataloader, concept_cols, device):
    """Extract features and train linear probes for each concept."""
    from sklearn.linear_model import LogisticRegression

    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            features = model.visual(imgs)
            if features.dim() == 3:
                features = features[:, 0, :]
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    probes = {}
    for i, concept in enumerate(concept_cols):
        y = labels[:, i]
        if y.sum() < 10 or (1 - y).sum() < 10:
            continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(features, y)
        acc = clf.score(features, y)
        probes[concept] = {
            "weight": clf.coef_[0].tolist(),
            "bias": float(clf.intercept_[0]),
            "train_accuracy": round(acc, 4),
        }

    return probes


# ---------------------------------------------------------------------------
# 5. Prepare sandbox
# ---------------------------------------------------------------------------

def prepare_environment(concept_df, img_dir, env_root, concept_cols,
                        forget_concepts, retain_concepts, model, probes,
                        model_type, n_images):
    """Create the agent's sandbox."""
    agent_data = env_root / "data"
    agent_images = agent_data / "images"
    agent_model = env_root / "model"
    agent_output = env_root / "output"
    judge_dir = env_root / ".judge"

    for d in [agent_data, agent_images, agent_model, agent_output, judge_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Limit and split
    n = min(n_images, len(concept_df))
    indices = np.random.RandomState(42).permutation(n)
    df = concept_df.iloc[indices[:n]].reset_index(drop=True)

    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    # Copy images for agent
    print("Copying train/val images...")
    for _, row in pd.concat([train_df, val_df]).iterrows():
        src = img_dir / row["filename"]
        dst = agent_images / row["filename"]
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    # Save labels
    train_df.to_csv(agent_data / "concept_labels_train.csv", index=False)
    val_df.to_csv(agent_data / "concept_labels_val.csv", index=False)

    # Save the fine-tuned model (this is what the agent must edit)
    torch.save({
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "num_concepts": len(concept_cols),
        "concept_cols": concept_cols,
    }, agent_model / "concept_classifier.pt")

    # Save linear probes (agent can use these to verify unlearning)
    with open(agent_model / "concept_probes.json", "w") as f:
        json.dump(probes, f)

    # Save config
    config = {
        "model_type": model_type,
        "model_arch": "ViT-B/16 + linear head",
        "num_layers": 12,
        "hidden_dim": 768,
        "num_concepts": len(concept_cols),
        "all_concepts": concept_cols,
        "forget_concepts": forget_concepts,
        "retain_concepts": retain_concepts,
        "n_train": n_train,
        "n_val": len(val_df),
        "image_size": 224,
        "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
        "normalize_std": [0.26862954, 0.26130258, 0.27577711],
    }
    with open(agent_model / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Judge: test images + labels + original model ---
    judge_images = judge_dir / "images"
    judge_images.mkdir(exist_ok=True)

    print("Copying test images (hidden)...")
    for _, row in test_df.iterrows():
        src = img_dir / row["filename"]
        dst = judge_images / row["filename"]
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    test_df.to_csv(judge_dir / "test_labels.csv", index=False)

    # Save original model for judge comparison
    torch.save({
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "num_concepts": len(concept_cols),
        "concept_cols": concept_cols,
    }, judge_dir / "original_model.pt")

    judge_config = {
        "n_test": len(test_df),
        "model_type": model_type,
        "concept_cols": concept_cols,
        "forget_concepts": forget_concepts,
        "retain_concepts": retain_concepts,
    }
    with open(judge_dir / "judge_config.json", "w") as f:
        json.dump(judge_config, f, indent=2)

    print(f"\nEnvironment ready at: {env_root}")
    print(f"  Forget concepts: {forget_concepts}")
    print(f"  Retain concepts: {len(retain_concepts)} concepts")
    print(f"  Train: {n_train}, Val: {len(val_df)}, Test: {len(test_df)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=15000)
    parser.add_argument("--n_forget", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    args = parser.parse_args()

    env_root = Path(args.env_root)
    raw_dir = env_root / ".raw"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Download
    print("=" * 50)
    print("Step 1: Downloading CelebA...")
    print("=" * 50)
    concept_df, img_dir, attr_names = download_celeba(raw_dir)
    concept_cols = attr_names
    print(f"  {len(concept_df)} images, {len(concept_cols)} concepts")

    # Step 2: Choose forget/retain
    print("=" * 50)
    print("Step 2: Choosing concepts to forget...")
    print("=" * 50)
    forget_concepts, retain_concepts = choose_concepts(concept_df, concept_cols, args.n_forget)
    print(f"  Forget: {forget_concepts}")
    print(f"  Retain: {len(retain_concepts)} concepts")

    # Step 3: Fine-tune CLIP classifier
    print("=" * 50)
    print("Step 3: Fine-tuning CLIP concept classifier...")
    print("=" * 50)

    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        visual = clip_model.visual
        model_type = "open_clip"
    except ImportError:
        import timm
        visual = timm.create_model("vit_base_patch16_224", pretrained=True)
        model_type = "timm"

    model = CLIPConceptClassifier(visual, len(concept_cols))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])

    n = min(args.n_images, len(concept_df))
    indices = np.random.RandomState(42).permutation(n)
    df = concept_df.iloc[indices[:n]].reset_index(drop=True)

    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]

    train_ds = CelebADataset(img_dir, train_df, concept_cols, transform)
    val_ds = CelebADataset(img_dir, val_df, concept_cols, transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    model = finetune_classifier(model, train_loader, val_loader, concept_cols,
                                epochs=args.finetune_epochs, device=device)

    # Step 4: Train probes
    print("=" * 50)
    print("Step 4: Training concept probes...")
    print("=" * 50)
    probes = train_probes(model, train_loader, concept_cols, device)
    print(f"  Trained {len(probes)} probes")

    # Step 5: Prepare sandbox
    print("=" * 50)
    print("Step 5: Preparing sandbox...")
    print("=" * 50)
    model = model.cpu()
    prepare_environment(
        concept_df, img_dir, env_root, concept_cols,
        forget_concepts, retain_concepts, model, probes,
        model_type, args.n_images
    )


if __name__ == "__main__":
    main()
