#!/usr/bin/env python
"""
aggregate_fold_results.py
============================
Reads the 5 cv_report_fold{N}_{pooling}.json files written by
06a_train_cnn_cv.py, and reports mean +/- std of overall balanced accuracy
and per-class recall/F1/AUC across the 5 CV folds.

This is metric averaging (option (a)), not ensembling: each of the 5 models
is a separate model evaluated on its own fold's test partition.
"""

import argparse
import json
import os

import pandas as pd


def load_all(model_dir, pooling):
    rows = []
    for fold in range(1, 6):
        path = os.path.join(model_dir, f"cv_report_fold{fold}_{pooling}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path} -- skipping (job may not have finished)")
            continue
        with open(path) as f:
            d = json.load(f)
        row = {
            "pooling": pooling,
            "fold": fold,
            "val_balanced_accuracy": d["val_balanced_accuracy"],
            "val_auc_macro": d.get("val_auc_macro"),
            "val_auc_weighted": d.get("val_auc_weighted"),
        }
        per_class = d.get("val_per_class_report", {})
        for cls_name, cls_metrics in per_class.items():
            if cls_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            row[f"recall_{cls_name}"] = cls_metrics["recall"]
            row[f"f1_{cls_name}"] = cls_metrics["f1-score"]
        auc_per_class = d.get("val_auc_per_class", {})
        for cls_name, auc_val in auc_per_class.items():
            row[f"auc_{cls_name}"] = auc_val
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="/home/sp00001/blood_mil_project/models/gen2_cnn")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--output_csv", default=None)
    args = p.parse_args()

    df = load_all(args.model_dir, args.pooling)
    if df.empty:
        print("No reports found -- have the 5 CV jobs finished?")
        return

    metric_cols = [c for c in df.columns if c not in ("pooling", "fold")]

    print("=== Per-fold raw values ===")
    print(df.sort_values("fold").to_string(index=False))

    print(f"\n=== Mean +/- std across 5 folds ({args.pooling} pooling) ===")
    row = {"pooling": args.pooling, "n_folds": len(df)}
    for col in metric_cols:
        mean_v = df[col].mean()
        std_v = df[col].std()
        print(f"  {col:<28} {mean_v:.3f} +/- {std_v:.3f}")
        row[f"{col}_mean"] = mean_v
        row[f"{col}_std"] = std_v

    summary_df = pd.DataFrame([row])
    out_csv = args.output_csv or os.path.join(args.model_dir, f"fold_aggregate_summary_{args.pooling}.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved aggregate summary to: {out_csv}")

    print("\n=== Classes with mean recall < 0.15 across folds ===")
    recall_cols = [c for c in metric_cols if c.startswith("recall_")]
    for col in recall_cols:
        m = df[col].mean()
        if m < 0.15:
            print(f"  {col.replace('recall_', '')}: mean recall = {m:.3f} across folds {sorted(df['fold'].tolist())}")


if __name__ == "__main__":
    main()
