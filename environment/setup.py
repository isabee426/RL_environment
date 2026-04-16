"""
Environment Setup: Transcoders on Vision Transformers
=====================================================
Prepares the sandbox for the agent:
1. Downloads a pretrained CLIP ViT-B/16
2. Downloads a subset of images with concept labels (from CelebA attributes)
3. Extracts MLP input/output activation pairs from a target layer
4. Trains a baseline SAE for comparison
5. Splits data: agent gets train/val activations + concept labels, judge keeps test set

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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------

def download_celeba_subset(raw_dir: Path):
    """Download CelebA attributes and a subset of images.

    CelebA has 40 binary attributes per face image — perfect for
    evaluating whether learned features correspond to visual concepts.
    We use a subset (20k images) to keep things manageable.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download CelebA attributes
    attr_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
    attr_path = raw_dir / "list_attr_celeba.txt"
    img_dir = raw_dir / "img_celeba"

    if not attr_path.exists():
        print("Downloading CelebA attributes...")
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
                str(attr_path), quiet=False
            )
        except Exception as e:
            print(f"gdown failed: {e}")
            print("Generating synthetic concept labels instead...")
            generate_synthetic_data(raw_dir)
            return raw_dir

    if not img_dir.exists() or len(list(img_dir.glob("*.jpg"))) < 1000:
        print("Downloading CelebA images (subset)...")
        try:
            import gdown
            zip_path = raw_dir / "img_celeba.zip"
            gdown.download(
                "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                str(zip_path), quiet=False
            )
            subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(raw_dir)], check=True)
        except Exception as e:
            print(f"CelebA download failed: {e}")
            print("Generating synthetic data with CIFAR-10 instead...")
            generate_synthetic_data(raw_dir)
            return raw_dir

    return raw_dir


def generate_synthetic_data(raw_dir: Path):
    """Fallback: use CIFAR-10 with class labels as 'concepts'."""
    from torchvision.datasets import CIFAR10

    print("Downloading CIFAR-10 as fallback dataset...")
    ds = CIFAR10(root=str(raw_dir), train=True, download=True)

    img_dir = raw_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Save first 20k images
    n_images = min(20000, len(ds))
    filenames = []
    labels = []

    for i in range(n_images):
        img, label = ds[i]
        fname = f"img_{i:06d}.jpg"
        img.save(img_dir / fname)
        filenames.append(fname)
        labels.append(label)

    # CIFAR-10 classes as binary concept columns
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    concept_data = {"filename": filenames}
    for j, cname in enumerate(class_names):
        concept_data[cname] = [1 if l == j else 0 for l in labels]

    df = pd.DataFrame(concept_data)
    df.to_csv(raw_dir / "concept_labels.csv", index=False)

    print(f"Saved {n_images} images and concept labels")
    return raw_dir


# ---------------------------------------------------------------------------
# 2. Extract activations from CLIP ViT
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, filenames, transform):
        self.img_dir = Path(img_dir)
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.filenames[idx]).convert("RGB")
        return self.transform(img), idx


def get_clip_model():
    """Load CLIP ViT-B/16 from open_clip or torchvision."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        return model.visual, preprocess, "open_clip"
    except ImportError:
        pass

    # Fallback: use timm's ViT-B/16 pretrained on ImageNet
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, preprocess, "timm"


def extract_mlp_activations(model, dataloader, target_layer: int, device, model_type: str):
    """Extract MLP input and output activation pairs from a target layer.

    For ViT-B/16: 12 transformer blocks, each with:
      - self-attention
      - MLP (fc1 -> GELU -> fc2)

    We hook the MLP to capture its input (post-layernorm) and output.
    """
    model = model.to(device)
    model.eval()

    mlp_inputs = []
    mlp_outputs = []

    # Register hooks based on model type
    if model_type == "open_clip":
        block = model.transformer.resblocks[target_layer]
        mlp_module = block.mlp
        ln_module = block.ln_2
    elif model_type == "timm":
        block = model.blocks[target_layer]
        mlp_module = block.mlp
        ln_module = block.norm2

    def hook_input(module, input, output):
        # Capture the input to the MLP (which is post-layernorm)
        mlp_inputs.append(input[0].detach().cpu())

    def hook_output(module, input, output):
        mlp_outputs.append(output.detach().cpu())

    handle_in = mlp_module.register_forward_hook(hook_input)
    handle_out = mlp_module.register_forward_hook(hook_output)

    print(f"Extracting activations from layer {target_layer}...")
    with torch.no_grad():
        for imgs, indices in dataloader:
            imgs = imgs.to(device)
            _ = model(imgs)

    handle_in.remove()
    handle_out.remove()

    # Concatenate: shape [n_images, n_tokens, hidden_dim]
    all_inputs = torch.cat(mlp_inputs, dim=0)
    all_outputs = torch.cat(mlp_outputs, dim=0)

    # Use CLS token only (token 0) for simplicity
    cls_inputs = all_inputs[:, 0, :]   # [n_images, hidden_dim]
    cls_outputs = all_outputs[:, 0, :]  # [n_images, hidden_dim]

    return cls_inputs.numpy(), cls_outputs.numpy()


# ---------------------------------------------------------------------------
# 3. Train baseline SAE
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """Standard SAE: reconstructs MLP output from MLP output."""
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.decoder.bias = nn.Parameter(torch.zeros(input_dim))
        # Tie decoder to mean-centered data

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def topk_forward(self, x, k):
        z_pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z_pre, k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, torch.relu(topk_vals))
        x_hat = self.decoder(z)
        return x_hat, z


def train_baseline_sae(mlp_outputs, dict_size=4096, k=32, epochs=20,
                       lr=1e-3, batch_size=256, device="cpu"):
    """Train a baseline SAE on MLP outputs using TopK sparsity."""
    print(f"Training baseline SAE (dict_size={dict_size}, k={k})...")

    data = torch.tensor(mlp_outputs, dtype=torch.float32)
    input_dim = data.shape[1]

    sae = SparseAutoencoder(input_dim, dict_size).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        total_l0 = 0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z = sae.topk_forward(batch, k)
            loss = ((batch - x_hat) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l0 += (z > 0).float().sum(dim=-1).mean().item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            avg_l0 = total_l0 / n_batches
            print(f"  Epoch {epoch+1}/{epochs} | MSE: {avg_loss:.6f} | L0: {avg_l0:.1f}")

    return sae


# ---------------------------------------------------------------------------
# 4. Prepare sandboxed environment
# ---------------------------------------------------------------------------

def prepare_environment(raw_dir: Path, env_root: Path, model_type: str,
                        target_layer: int, train_inputs: np.ndarray,
                        train_outputs: np.ndarray, val_inputs: np.ndarray,
                        val_outputs: np.ndarray, test_inputs: np.ndarray,
                        test_outputs: np.ndarray, baseline_sae: nn.Module,
                        concept_labels_train: pd.DataFrame,
                        concept_labels_val: pd.DataFrame,
                        concept_labels_test: pd.DataFrame):
    """Create the agent's sandboxed workspace."""

    agent_data = env_root / "data"
    agent_model = env_root / "model"
    agent_output = env_root / "output"
    judge_dir = env_root / ".judge"

    for d in [agent_data, agent_model, agent_output, judge_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Agent gets train + val activations ---
    np.save(agent_data / "mlp_inputs_train.npy", train_inputs)
    np.save(agent_data / "mlp_outputs_train.npy", train_outputs)
    np.save(agent_data / "mlp_inputs_val.npy", val_inputs)
    np.save(agent_data / "mlp_outputs_val.npy", val_outputs)
    concept_labels_train.to_csv(agent_data / "concept_labels_train.csv", index=False)
    concept_labels_val.to_csv(agent_data / "concept_labels_val.csv", index=False)

    # --- Agent gets the baseline SAE for comparison ---
    torch.save({
        "state_dict": baseline_sae.state_dict(),
        "input_dim": train_outputs.shape[1],
        "dict_size": baseline_sae.encoder.out_features,
    }, agent_model / "baseline_sae.pt")

    # --- Model config ---
    config = {
        "source_model": "ViT-B/16",
        "model_type": model_type,
        "target_layer": target_layer,
        "hidden_dim": int(train_inputs.shape[1]),
        "mlp_output_dim": int(train_outputs.shape[1]),
        "n_train": int(train_inputs.shape[0]),
        "n_val": int(val_inputs.shape[0]),
        "concepts": [c for c in concept_labels_train.columns if c != "filename"],
        "baseline_sae": {
            "dict_size": baseline_sae.encoder.out_features,
            "topk": 32,
        },
    }
    with open(agent_model / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Judge gets test set ---
    np.save(judge_dir / "mlp_inputs_test.npy", test_inputs)
    np.save(judge_dir / "mlp_outputs_test.npy", test_outputs)
    concept_labels_test.to_csv(judge_dir / "concept_labels_test.csv", index=False)

    # Also save baseline SAE metrics on test set for comparison
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_sae = baseline_sae.to(device)
    test_tensor = torch.tensor(test_outputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_recon, test_z = baseline_sae.topk_forward(test_tensor, k=32)
        sae_mse = ((test_tensor - test_recon) ** 2).mean().item()
        sae_l0 = (test_z > 0).float().sum(dim=-1).mean().item()

    # Compute per-feature activations for interpretability scoring
    test_activations = test_z.cpu().numpy()  # [n_test, dict_size]
    np.save(judge_dir / "baseline_sae_activations.npy", test_activations)

    judge_meta = {
        "baseline_sae_mse": sae_mse,
        "baseline_sae_l0": sae_l0,
    }
    with open(judge_dir / "judge_meta.json", "w") as f:
        json.dump(judge_meta, f, indent=2)

    print(f"\nEnvironment ready at: {env_root}")
    print(f"  Agent data: {agent_data}")
    print(f"  Agent model: {agent_model}")
    print(f"  Judge data: {judge_dir} (hidden)")
    print(f"  Baseline SAE MSE: {sae_mse:.6f}, L0: {sae_l0:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_root", type=str, required=True)
    parser.add_argument("--target_layer", type=int, default=8,
                        help="Which ViT layer to extract MLP activations from (0-11)")
    parser.add_argument("--n_images", type=int, default=15000,
                        help="Number of images to use")
    parser.add_argument("--dict_size", type=int, default=4096,
                        help="Baseline SAE dictionary size")
    parser.add_argument("--sae_epochs", type=int, default=20)
    args = parser.parse_args()

    env_root = Path(args.env_root)
    raw_dir = env_root / ".raw"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Download data
    print("=" * 50)
    print("Step 1: Downloading data...")
    print("=" * 50)
    data_dir = download_celeba_subset(raw_dir)

    # Find images
    concept_csv = raw_dir / "concept_labels.csv"
    if concept_csv.exists():
        concept_df = pd.read_csv(concept_csv)
        img_dir = raw_dir / "images"
    else:
        # Parse CelebA attribute file
        attr_path = raw_dir / "list_attr_celeba.txt"
        # CelebA zip extracts to img_align_celeba/ not img_celeba/
        img_dir = raw_dir / "img_align_celeba"
        if not img_dir.exists():
            img_dir = raw_dir / "img_celeba"

        with open(attr_path) as f:
            n_images = int(f.readline().strip())
            attr_names = f.readline().strip().split()
            rows = []
            for line in f:
                parts = line.strip().split()
                fname = parts[0]
                attrs = [1 if int(x) == 1 else 0 for x in parts[1:]]
                rows.append([fname] + attrs)

        concept_df = pd.DataFrame(rows, columns=["filename"] + attr_names)

    # Limit to n_images
    n = min(args.n_images, len(concept_df))
    concept_df = concept_df.iloc[:n].reset_index(drop=True)
    filenames = concept_df["filename"].tolist()

    # Step 2: Load model and extract activations
    print("=" * 50)
    print("Step 2: Loading model and extracting activations...")
    print("=" * 50)
    model, preprocess, model_type = get_clip_model()

    dataset = ImageFolderDataset(img_dir, filenames, preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mlp_inputs, mlp_outputs = extract_mlp_activations(
        model, loader, args.target_layer, device, model_type
    )
    print(f"  MLP inputs shape: {mlp_inputs.shape}")
    print(f"  MLP outputs shape: {mlp_outputs.shape}")

    # Step 3: Train/val/test split (60/20/20)
    print("=" * 50)
    print("Step 3: Splitting data...")
    print("=" * 50)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_in, train_out = mlp_inputs[train_idx], mlp_outputs[train_idx]
    val_in, val_out = mlp_inputs[val_idx], mlp_outputs[val_idx]
    test_in, test_out = mlp_inputs[test_idx], mlp_outputs[test_idx]

    concept_cols = [c for c in concept_df.columns if c != "filename"]
    concept_train = concept_df.iloc[train_idx][concept_cols].reset_index(drop=True)
    concept_val = concept_df.iloc[val_idx][concept_cols].reset_index(drop=True)
    concept_test = concept_df.iloc[test_idx][concept_cols].reset_index(drop=True)

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Step 4: Train baseline SAE
    print("=" * 50)
    print("Step 4: Training baseline SAE...")
    print("=" * 50)
    baseline_sae = train_baseline_sae(
        train_out, dict_size=args.dict_size, k=32,
        epochs=args.sae_epochs, device=device
    )

    # Step 5: Prepare sandbox
    print("=" * 50)
    print("Step 5: Preparing sandbox...")
    print("=" * 50)
    prepare_environment(
        raw_dir, env_root, model_type, args.target_layer,
        train_in, train_out, val_in, val_out, test_in, test_out,
        baseline_sae, concept_train, concept_val, concept_test
    )


if __name__ == "__main__":
    main()
