#!/usr/bin/env python
"""
aggregate_fold_results.py
============================
Reads all holdout_report_fold{N}_{pooling}.json files (N=1..5) written by the
5-fold x 3-pooling array job, and for EACH pooling, reports:
  - mean +/- std of overall holdout balanced accuracy across the 5 folds
  - mean +/- std of per-class recall / F1 across the 5 folds (this is what
    lets you check whether the CBFB_MYH11 recall=0 pattern seen in fold 1
    holds up across all 5 folds, or was fold-1-specific)

This is metric averaging (option (a)), NOT ensembling: each of the 5 models
stays a separate model evaluated independently on the same holdout set: no
single "final" model comes out of this, just a distribution of how well this
(pooling, architecture, hyperparameter) configuration tends to generalize.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

POOLINGS = ["mean", "max", "min_max"]


def load_all(model_dir, split="val"):
    """split='val': the fold's own test partition -- now the OFFICIAL final
    metric per team decision (holdout_data_for_multiclass is currently unused).
    split='holdout': the old fixed 28-patient holdout, kept only for reference."""
    prefix = "val" if split == "val" else "holdout"
    bal_acc_key = "best_val_balanced_accuracy" if split == "val" else "holdout_balanced_accuracy"
    report_key = f"{prefix}_per_class_report"
    auc_macro_key = f"{prefix}_auc_macro"
    auc_weighted_key = f"{prefix}_auc_weighted"
    auc_per_class_key = f"{prefix}_auc_per_class"

    rows = []
    for pooling in POOLINGS:
        for fold in range(1, 6):
            path = os.path.join(model_dir, f"holdout_report_fold{fold}_{pooling}.json")
            if not os.path.exists(path):
                print(f"WARNING: missing {path} -- skipping (job may not have finished)")
                continue
            with open(path) as f:
                d = json.load(f)
            row = {
                "pooling": pooling,
                "fold": fold,
                f"{prefix}_balanced_accuracy": d[bal_acc_key],
                f"{prefix}_auc_macro": d.get(auc_macro_key),
                f"{prefix}_auc_weighted": d.get(auc_weighted_key),
            }
            per_class = d.get(report_key, {})
            for cls_name, cls_metrics in per_class.items():
                if cls_name in ("accuracy", "macro avg", "weighted avg"):
                    continue
                row[f"recall_{cls_name}"] = cls_metrics["recall"]
                row[f"f1_{cls_name}"] = cls_metrics["f1-score"]
            auc_per_class = d.get(auc_per_class_key, {})
            for cls_name, auc_val in auc_per_class.items():
                row[f"auc_{cls_name}"] = auc_val
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="/home/sp00001/blood_mil_project/models/gen2_cnn")
    p.add_argument("--split", choices=["val", "holdout"], default="val",
                   help="val = each fold's own test partition (now the OFFICIAL metric "
                        "per team decision). holdout = old fixed 28-patient set, reference only.")
    p.add_argument("--output_csv", default=None)
    args = p.parse_args()

    df = load_all(args.model_dir, split=args.split)
    if df.empty:
        print("No reports found -- have the 15 jobs finished?")
        return
    print(f"(using split='{args.split}' as the metric source)\n")

    metric_cols = [c for c in df.columns if c not in ("pooling", "fold")]

    print("=== Per-fold raw values ===")
    print(df.sort_values(["pooling", "fold"]).to_string(index=False))

    print("\n=== Mean +/- std across 5 folds, by pooling ===")
    summary_rows = []
    for pooling, group in df.groupby("pooling"):
        print(f"\n{pooling}  (n_folds={len(group)}):")
        row = {"pooling": pooling, "n_folds": len(group)}
        for col in metric_cols:
            mean_v = group[col].mean()
            std_v = group[col].std()
            print(f"  {col:<28} {mean_v:.3f} +/- {std_v:.3f}")
            row[f"{col}_mean"] = mean_v
            row[f"{col}_std"] = std_v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_csv = args.output_csv or os.path.join(args.model_dir, "fold_aggregate_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved aggregate summary to: {out_csv}")

    # Flag classes with consistently near-zero recall across folds -- this is
    # the check for whether the CBFB_MYH11 collapse seen in fold 1 generalizes.
    print("\n=== Classes with mean recall < 0.15 across folds (per pooling) ===")
    for pooling, group in df.groupby("pooling"):
        recall_cols = [c for c in metric_cols if c.startswith("recall_")]
        for col in recall_cols:
            m = group[col].mean()
            if m < 0.15:
                print(f"  {pooling} / {col.replace('recall_', '')}: "
                      f"mean recall = {m:.3f} across folds {sorted(group['fold'].tolist())}")


if __name__ == "__main__":
    main()
