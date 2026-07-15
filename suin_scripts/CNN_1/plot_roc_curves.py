#!/usr/bin/env python
"""
plot_roc_curves.py
=====================
One subplot per class (one-vs-rest ROC). Each subplot shows:
  - 5 thin CV fold ROC curves (from 06a_train_cnn_cv.py's cv_roc_fold{N}_{pooling}.json)
  - 1 bold mean CV ROC curve (fpr grid + averaged/interpolated tpr, standard
    sklearn "ROC with cross validation" technique)
  - 1 dashed holdout ROC curve (from 06b's final_holdout_roc_final_{pooling}.json)

Usage:
    python plot_roc_curves.py --model_dir .../gen2_cnn --pooling mean
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_cv_roc(model_dir, pooling):
    """Returns {class_name: [ (fpr, tpr) for each of 5 folds ]}"""
    per_class = {}
    for fold in range(1, 6):
        path = os.path.join(model_dir, f"cv_roc_fold{fold}_{pooling}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        for cls_name, roc in d.items():
            per_class.setdefault(cls_name, []).append(roc)  # roc may be None
    return per_class


def load_holdout_roc(model_dir, pooling):
    path = os.path.join(model_dir, f"final_holdout_roc_final_{pooling}.json")
    with open(path) as f:
        return json.load(f)


def mean_roc(fold_rocs, n_points=100):
    """Interpolate each fold's TPR onto a common FPR grid, then average."""
    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []
    for roc in fold_rocs:
        if roc is None:
            continue
        fpr, tpr = np.array(roc["fpr"]), np.array(roc["tpr"])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    if not tprs:
        return mean_fpr, None
    return mean_fpr, np.mean(tprs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cv_roc = load_cv_roc(args.model_dir, args.pooling)
    holdout_roc = load_holdout_roc(args.model_dir, args.pooling)
    classes = list(cv_roc.keys())

    fig, axes = plt.subplots(1, len(classes), figsize=(4.5 * len(classes), 4.5))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls_name in zip(axes, classes):
        fold_rocs = cv_roc.get(cls_name, [])
        for i, roc in enumerate(fold_rocs):
            if roc is None:
                continue
            ax.plot(roc["fpr"], roc["tpr"], color="tab:blue", alpha=0.3, lw=1,
                    label="CV fold" if i == 0 else None)

        mean_fpr, mean_tpr = mean_roc(fold_rocs)
        if mean_tpr is not None:
            ax.plot(mean_fpr, mean_tpr, color="tab:blue", lw=2.5, label="CV mean")

        hold = holdout_roc.get(cls_name)
        if hold is not None:
            ax.plot(hold["fpr"], hold["tpr"], color="tab:red", lw=2, linestyle="--", label="Holdout (final)")

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
        ax.set_title(cls_name, fontsize=10)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"gen2_cnn ROC curves ({args.pooling} pooling) -- CV folds vs holdout")
    fig.tight_layout()

    out_path = args.output or os.path.join(args.model_dir, f"roc_curves_{args.pooling}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
