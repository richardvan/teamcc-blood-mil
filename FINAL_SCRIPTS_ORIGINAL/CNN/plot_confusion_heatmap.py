#!/usr/bin/env python
"""
plot_confusion_heatmap.py
============================
Confusion matrix heatmap for a gen2_cnn holdout result (matplotlib only, no
seaborn dependency). Can plot a single (fold, pooling) run, or the SUMMED
confusion matrix across all 5 folds for one pooling (recommended for the
presentation -- one fold alone is noisy given ~28 holdout patients).

Usage:
    # single fold
    python plot_confusion_heatmap.py --model_dir .../gen2_cnn --pooling max --fold 3

    # summed across all 5 folds (recommended)
    python plot_confusion_heatmap.py --model_dir .../gen2_cnn --pooling max --all_folds
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np


def load_cm(model_dir, pooling, fold=None, split="val"):
    """split='val': CV fold's own test partition, from 06a's cv_report_fold{N}_{pooling}.json.
    split='holdout': the one-time final holdout result, from 06b's final_holdout_report_final_{pooling}.json
    (fold is ignored for holdout, since there's only one final model)."""
    if split == "holdout":
        path = os.path.join(model_dir, f"final_holdout_report_final_{pooling}.json")
        with open(path) as f:
            d = json.load(f)
        return np.array(d["holdout_confusion_matrix"]), d["class_order"]
    else:
        path = os.path.join(model_dir, f"cv_report_fold{fold}_{pooling}.json")
        with open(path) as f:
            d = json.load(f)
        return np.array(d["val_confusion_matrix"]), d["class_order"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pooling", required=True, choices=["mean", "max", "min", "min_max"])
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--all_folds", action="store_true",
                   help="Sum the confusion matrix across all 5 folds instead of one.")
    p.add_argument("--split", choices=["val", "holdout"], default="val",
                   help="val = fold's own test partition (now official). holdout = old fixed set.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.split == "holdout":
        cm, classes = load_cm(args.model_dir, args.pooling, split="holdout")
        title = f"gen2_cnn ({args.pooling} pooling) -- holdout (final, one-time)"
    elif args.all_folds:
        cm_total, classes = None, None
        for fold in range(1, 6):
            cm, classes = load_cm(args.model_dir, args.pooling, fold, args.split)
            cm_total = cm if cm_total is None else cm_total + cm
        cm, title = cm_total, f"gen2_cnn ({args.pooling} pooling) -- {args.split}, summed over 5 folds"
    else:
        if args.fold is None:
            raise ValueError("Provide --fold N or use --all_folds")
        cm, classes = load_cm(args.model_dir, args.pooling, args.fold, args.split)
        title = f"gen2_cnn ({args.pooling} pooling) -- {args.split}, fold {args.fold}"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate each cell with its count
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()

    if args.split == "holdout":
        suffix = "final"
    elif args.all_folds:
        suffix = "all_folds"
    else:
        suffix = f"fold{args.fold}"

    out_path = args.output or os.path.join(
        args.model_dir, f"confusion_heatmap_{args.pooling}_{args.split}_{suffix}.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
