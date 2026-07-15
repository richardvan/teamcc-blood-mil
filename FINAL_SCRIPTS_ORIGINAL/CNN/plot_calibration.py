#!/usr/bin/env python
"""
plot_calibration.py
=======================
Calibration (reliability) plot: for each class, does a predicted probability
of e.g. 0.7 actually correspond to being right ~70% of the time? All 5 classes
overlaid in ONE panel (unlike ROC/PR, calibration curves don't overlap as
confusingly, so a single combined plot per split is standard).

Only 5 quantile-based bins are used (--n_calibration_bins in
compute_extra_metrics.py) because CV val folds have ~30 patients and holdout
has 28 -- 10+ bins would leave many bins with 0-1 patients, producing a
meaningless jagged line.

Usage:
    python plot_calibration.py --model_dir .../gen2_cnn --pooling mean
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "CBFB_MYH11": "tab:blue",
    "NPM1": "tab:green",
    "PML_RARA": "tab:brown",
    "RUNX1_RUNX1T1": "tab:gray",
    "control": "tab:cyan",
}


def plot_split(cal_d, title, output):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="perfectly calibrated")

    for cls_name, color in CLASS_COLORS.items():
        cal = cal_d.get(cls_name)
        if cal is None:
            continue
        ax.plot(cal["mean_predicted"], cal["observed_freq"], marker="o", color=color, label=cls_name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Where calibration_*.json results live.")
    p.add_argument("--output_dir", default=None, help="Where to save PNGs (default: same as model_dir).")
    p.add_argument("--pooling", default="mean")
    args = p.parse_args()
    out_dir = args.output_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    cv_path = os.path.join(args.model_dir, f"calibration_fold1_{args.pooling}.json")
    if os.path.exists(cv_path):
        with open(cv_path) as f:
            cv_cal = json.load(f)
        plot_split(cv_cal, f"Calibration (CV fold 1, illustrative) -- {args.pooling} pooling",
                   os.path.join(out_dir, f"calibration_cv_{args.pooling}.png"))
    else:
        print(f"WARNING: missing {cv_path}")

    holdout_path = os.path.join(args.model_dir, f"calibration_holdout_final_{args.pooling}.json")
    if os.path.exists(holdout_path):
        with open(holdout_path) as f:
            holdout_cal = json.load(f)
        plot_split(holdout_cal, f"Calibration (Holdout, final model) -- {args.pooling} pooling",
                   os.path.join(out_dir, f"calibration_holdout_{args.pooling}.png"))
    else:
        print(f"WARNING: missing {holdout_path}")


if __name__ == "__main__":
    main()
