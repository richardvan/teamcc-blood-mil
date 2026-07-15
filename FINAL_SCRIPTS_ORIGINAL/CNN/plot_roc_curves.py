#!/usr/bin/env python
"""
plot_roc_curves.py
=====================
Matches the SVM teammate's ROC plot style: a 2x3 grid per figure --
top-left panel overlays all 5 classes' mean ROC curves together (with
AUC +/- std in the legend), and the remaining 5 panels show each class
individually.

Produces TWO SEPARATE figures (per the team's request):
  - roc_curves_cv_{pooling}.png       : CV/train side. Each class panel shows
                                         5 faint per-fold curves + 1 bold mean
                                         curve (from 06a's cv_roc_fold{N}_{pooling}.json).
                                         Legend shows AUC mean +/- std across folds.
  - roc_curves_holdout_{pooling}.png  : Holdout side. Each class panel shows
                                         ONE curve (the final, one-time model,
                                         from 06b's final_holdout_roc_final_{pooling}.json).
                                         Legend shows a single AUC value (no
                                         +/- std, since there's only one holdout
                                         evaluation by design).

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

CLASS_COLORS = {
    "CBFB_MYH11": "tab:blue",
    "NPM1": "tab:green",
    "PML_RARA": "tab:brown",
    "RUNX1_RUNX1T1": "tab:gray",
    "control": "tab:cyan",
}


def load_cv_roc_and_auc(model_dir, pooling):
    """Returns {class_name: [(fpr, tpr) per fold]}, {class_name: [auc per fold]}"""
    roc_per_class, auc_per_class = {}, {}
    for fold in range(1, 6):
        roc_path = os.path.join(model_dir, f"cv_roc_fold{fold}_{pooling}.json")
        report_path = os.path.join(model_dir, f"cv_report_fold{fold}_{pooling}.json")
        if not os.path.exists(roc_path) or not os.path.exists(report_path):
            print(f"WARNING: missing fold {fold} files")
            continue
        with open(roc_path) as f:
            roc_d = json.load(f)
        with open(report_path) as f:
            report_d = json.load(f)
        for cls_name, roc in roc_d.items():
            roc_per_class.setdefault(cls_name, []).append(roc)
        for cls_name, auc_val in report_d.get("val_auc_per_class", {}).items():
            auc_per_class.setdefault(cls_name, []).append(auc_val)
    return roc_per_class, auc_per_class


def load_holdout_roc_and_auc(model_dir, pooling):
    roc_path = os.path.join(model_dir, f"final_holdout_roc_final_{pooling}.json")
    report_path = os.path.join(model_dir, f"final_holdout_report_final_{pooling}.json")
    with open(roc_path) as f:
        roc_d = json.load(f)
    with open(report_path) as f:
        report_d = json.load(f)
    return roc_d, report_d.get("holdout_auc_per_class", {})


def mean_roc(fold_rocs, n_points=100):
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


def plot_cv_figure(model_dir, pooling, output):
    roc_per_class, auc_per_class = load_cv_roc_and_auc(model_dir, pooling)
    classes = [c for c in CLASS_COLORS if c in roc_per_class]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    combined_ax = axes_flat[0]
    combined_ax.set_title("All subtypes (mean ROC)")

    for i, cls_name in enumerate(classes):
        ax = axes_flat[i + 1]
        color = CLASS_COLORS[cls_name]
        fold_rocs = roc_per_class.get(cls_name, [])
        for roc in fold_rocs:
            if roc is None:
                continue
            ax.plot(roc["fpr"], roc["tpr"], color=color, alpha=0.25, lw=1)

        mean_fpr, mean_tpr = mean_roc(fold_rocs)
        aucs = [a for a in auc_per_class.get(cls_name, []) if a is not None]
        auc_mean, auc_std = (np.mean(aucs), np.std(aucs)) if aucs else (None, None)
        label = f"mean (AUC={auc_mean:.3f} \u00b1 {auc_std:.3f})" if auc_mean is not None else "mean"

        if mean_tpr is not None:
            ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=label)
            combined_ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                              label=f"{cls_name} (AUC={auc_mean:.3f} \u00b1 {auc_std:.3f})")

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_title(cls_name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")

    combined_ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="chance")
    combined_ax.set_xlabel("False Positive Rate")
    combined_ax.set_ylabel("True Positive Rate")
    combined_ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"ROC curves (CV, per-fold + mean per subtype) -- {pooling} pooling")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


def plot_holdout_figure(model_dir, pooling, output):
    roc_d, auc_per_class = load_holdout_roc_and_auc(model_dir, pooling)
    classes = [c for c in CLASS_COLORS if c in roc_d]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    combined_ax = axes_flat[0]
    combined_ax.set_title("All subtypes (holdout ROC)")

    for i, cls_name in enumerate(classes):
        ax = axes_flat[i + 1]
        color = CLASS_COLORS[cls_name]
        roc = roc_d.get(cls_name)
        auc_val = auc_per_class.get(cls_name)
        label = f"holdout (AUC={auc_val:.3f})" if auc_val is not None else "holdout"

        if roc is not None:
            ax.plot(roc["fpr"], roc["tpr"], color=color, lw=2.5, label=label)
            combined_ax.plot(roc["fpr"], roc["tpr"], color=color, lw=2,
                              label=f"{cls_name} (AUC={auc_val:.3f})" if auc_val is not None else cls_name)

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_title(cls_name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")

    combined_ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="chance")
    combined_ax.set_xlabel("False Positive Rate")
    combined_ax.set_ylabel("True Positive Rate")
    combined_ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"ROC curves (Holdout, final one-time model) -- {pooling} pooling")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Where the CV/holdout JSON results live.")
    p.add_argument("--output_dir", default=None, help="Where to save PNGs (default: same as model_dir).")
    p.add_argument("--pooling", default="mean")
    args = p.parse_args()
    out_dir = args.output_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    cv_out = os.path.join(out_dir, f"roc_curves_cv_{args.pooling}.png")
    holdout_out = os.path.join(out_dir, f"roc_curves_holdout_{args.pooling}.png")

    plot_cv_figure(args.model_dir, args.pooling, cv_out)
    plot_holdout_figure(args.model_dir, args.pooling, holdout_out)


if __name__ == "__main__":
    main()
