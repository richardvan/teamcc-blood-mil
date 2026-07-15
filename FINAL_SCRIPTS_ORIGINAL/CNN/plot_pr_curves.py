#!/usr/bin/env python
"""
plot_pr_curves.py
====================
Precision-Recall curves, same 6-panel style as plot_roc_curves.py (combined
overview panel + one panel per class), split into a CV figure and a holdout
figure. More informative than ROC for the rare classes (PML_RARA etc.) since
PR curves don't get inflated by the large number of true negatives.

Usage:
    python plot_pr_curves.py --model_dir .../gen2_cnn --pooling mean
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = {
    "CBFB_MYH11": "tab:blue",
    "NPM1": "tab:green",
    "PML_RARA": "tab:brown",
    "RUNX1_RUNX1T1": "tab:gray",
    "control": "tab:cyan",
}


def load_cv_pr(model_dir, pooling):
    pr_per_class = {}
    for fold in range(1, 6):
        path = os.path.join(model_dir, f"pr_fold{fold}_{pooling}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        for cls_name, pr in d.items():
            pr_per_class.setdefault(cls_name, []).append(pr)
    return pr_per_class


def load_holdout_pr(model_dir, pooling):
    path = os.path.join(model_dir, f"pr_holdout_final_{pooling}.json")
    with open(path) as f:
        return json.load(f)


def plot_cv_figure(model_dir, pooling, output):
    pr_per_class = load_cv_pr(model_dir, pooling)
    classes = [c for c in CLASS_COLORS if c in pr_per_class]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    combined_ax = axes_flat[0]
    combined_ax.set_title("All subtypes (PR curves)")

    for i, cls_name in enumerate(classes):
        ax = axes_flat[i + 1]
        color = CLASS_COLORS[cls_name]
        fold_prs = pr_per_class.get(cls_name, [])
        for pr in fold_prs:
            if pr is None:
                continue
            ax.plot(pr["recall"], pr["precision"], color=color, alpha=0.25, lw=1)

        # average precision at common recall grid (simple interpolation)
        common_recall = np.linspace(0, 1, 100)
        precisions = []
        for pr in fold_prs:
            if pr is None:
                continue
            recall, precision = np.array(pr["recall"]), np.array(pr["precision"])
            order = np.argsort(recall)
            interp_p = np.interp(common_recall, recall[order], precision[order])
            precisions.append(interp_p)
        if precisions:
            mean_p = np.mean(precisions, axis=0)
            ax.plot(common_recall, mean_p, color=color, lw=2.5, label="mean")
            combined_ax.plot(common_recall, mean_p, color=color, lw=2, label=cls_name)

        ax.set_title(cls_name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower left")

    combined_ax.set_xlabel("Recall")
    combined_ax.set_ylabel("Precision")
    combined_ax.set_xlim(0, 1)
    combined_ax.set_ylim(0, 1.02)
    combined_ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Precision-Recall curves (CV, per-fold + mean per subtype) -- {pooling} pooling")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


def plot_holdout_figure(model_dir, pooling, output):
    pr_d = load_holdout_pr(model_dir, pooling)
    classes = [c for c in CLASS_COLORS if c in pr_d]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    combined_ax = axes_flat[0]
    combined_ax.set_title("All subtypes (holdout PR)")

    for i, cls_name in enumerate(classes):
        ax = axes_flat[i + 1]
        color = CLASS_COLORS[cls_name]
        pr = pr_d.get(cls_name)
        if pr is not None:
            ax.plot(pr["recall"], pr["precision"], color=color, lw=2.5, label="holdout")
            combined_ax.plot(pr["recall"], pr["precision"], color=color, lw=2, label=cls_name)

        ax.set_title(cls_name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower left")

    combined_ax.set_xlabel("Recall")
    combined_ax.set_ylabel("Precision")
    combined_ax.set_xlim(0, 1)
    combined_ax.set_ylim(0, 1.02)
    combined_ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Precision-Recall curves (Holdout, final one-time model) -- {pooling} pooling")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Where pr_fold*/pr_holdout* JSON results live.")
    p.add_argument("--output_dir", default=None, help="Where to save PNGs (default: same as model_dir).")
    p.add_argument("--pooling", default="mean")
    args = p.parse_args()
    out_dir = args.output_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    cv_out = os.path.join(out_dir, f"pr_curves_cv_{args.pooling}.png")
    holdout_out = os.path.join(out_dir, f"pr_curves_holdout_{args.pooling}.png")

    plot_cv_figure(args.model_dir, args.pooling, cv_out)
    plot_holdout_figure(args.model_dir, args.pooling, holdout_out)


if __name__ == "__main__":
    main()
