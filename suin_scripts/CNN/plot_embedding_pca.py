#!/usr/bin/env python
"""
plot_embedding_pca.py
========================
Projects gen2_cnn's holdout bag-level features (post-pooling, pre-classifier)
to 2D via PCA (or t-SNE) and scatter-plots them colored by true subtype --
directly comparable to your friend's SVM pca_plot.png, since both show "how
separable are the classes in feature space" for their respective feature
representations (frozen SVM features vs fine-tuned CNN features).

Usage:
    python plot_embedding_pca.py \
        --checkpoint .../cnn_mil_fold1_max.pt \
        --organized_dir .../organized_data \
        --holdout_dir .../holdout_data_for_multiclass \
        --method pca
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import models, transforms

CLASSES_FALLBACK = ["control", "CBFB_MYH11", "NPM1", "PML_RARA", "RUNX1_RUNX1T1"]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_label(folder_name):
    parts = folder_name.split(".")
    return "control" if parts[0] == "normal" else parts[1]


def pool_features(feats, pooling):
    if pooling == "mean":
        return feats.mean(dim=0, keepdim=True)
    if pooling == "max":
        return feats.max(dim=0, keepdim=True).values
    if pooling == "min":
        return feats.min(dim=0, keepdim=True).values
    if pooling == "min_max":
        mn = feats.min(dim=0, keepdim=True).values
        mx = feats.max(dim=0, keepdim=True).values
        return torch.cat([mn, mx], dim=1)
    raise ValueError(pooling)


@torch.no_grad()
def get_bag_feature(backbone, pooling, image_paths, device, batch_size=64):
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack([PREPROCESS(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        feats.append(backbone(imgs))
    feats = torch.cat(feats, dim=0)
    return pool_features(feats, pooling).squeeze(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--holdout_dir", required=True)
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--method", choices=["pca", "tsne"], default="pca")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    pooling = ckpt.get("pooling", "mean")
    classes = ckpt.get("classes", CLASSES_FALLBACK)

    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()
    full_sd = ckpt["state_dict"]
    backbone_sd = {k[len("backbone."):]: v for k, v in full_sd.items() if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_sd)
    backbone.to(device).eval()

    holdout_file = os.path.join(args.holdout_dir, "holdout_patients.txt")
    with open(holdout_file) as f:
        holdout_patients = [line.strip() for line in f if line.strip()]

    features, labels_list = [], []
    for folder in holdout_patients:
        paths = sorted(glob.glob(os.path.join(args.organized_dir, folder, f"*{args.image_ext}")))
        if not paths:
            print(f"WARNING: no images for {folder}, skipping")
            continue
        feat = get_bag_feature(backbone, pooling, paths, device)
        features.append(feat)
        labels_list.append(parse_label(folder))

    X = np.stack(features)
    print(f"Feature matrix: {X.shape} ({len(set(labels_list))} classes)")

    if args.method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_2d = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = sorted(set(labels_list))
    cmap = plt.get_cmap("tab10")
    for i, cls_name in enumerate(unique_labels):
        mask = [l == cls_name for l in labels_list]
        pts = X_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=cls_name, color=cmap(i), s=60, alpha=0.8, edgecolors="k")

    method_label = "PCA" if args.method == "pca" else "t-SNE"
    ax.set_title(f"gen2_cnn holdout bag embeddings ({pooling} pooling) -- {method_label}")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend()
    fig.tight_layout()

    out_path = args.output or f"embedding_{args.method}_{pooling}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if args.method == "pca":
        var_ratio = reducer.explained_variance_ratio_
        print(f"Explained variance: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}, "
              f"total={var_ratio.sum():.2%}")
        print("Note: unlike your friend's SVM, this projects the FINE-TUNED CNN's "
              "features. If classes look more/less separated here than in their "
              "plot, that's a direct visual for the fine-tuning-vs-frozen comparison.")


if __name__ == "__main__":
    main()
