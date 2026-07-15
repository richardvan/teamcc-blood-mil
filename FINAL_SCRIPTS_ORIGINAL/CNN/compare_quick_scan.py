#!/usr/bin/env python
"""
compare_quick_scan.py
=========================
Compares the 6 (optimizer x unfreeze_from) combos from run_quick_scan_fold1.sh,
using fold 1 only. EXPLORATORY ONLY -- a single fold's result is noisy (as
we've seen throughout this project), so use this to narrow down candidates,
not to declare a final winner.

Usage:
    python compare_quick_scan.py --model_dir .../gen2_cnn
"""

import argparse
import json
import os

import pandas as pd

OPTIMIZERS = ["sgd", "adam"]
UNFREEZE = ["layer4", "layer3", "all"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    args = p.parse_args()

    rows = []
    for opt in OPTIMIZERS:
        for uf in UNFREEZE:
            path = os.path.join(args.model_dir, f"cv_report_fold1_mean_{opt}_{uf}.json")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}")
                continue
            with open(path) as f:
                d = json.load(f)
            row = {
                "optimizer": opt,
                "unfreeze_from": uf,
                "val_balanced_accuracy": d["val_balanced_accuracy"],
                "val_auc_macro": d.get("val_auc_macro"),
            }
            for cls_name, m in d.get("val_per_class_report", {}).items():
                if cls_name in ("accuracy", "macro avg", "weighted avg"):
                    continue
                row[f"recall_{cls_name}"] = m["recall"]
            rows.append(row)

    if not rows:
        print("No results found -- have the 6 quick-scan jobs finished?")
        return

    df = pd.DataFrame(rows).sort_values("val_balanced_accuracy", ascending=False)
    print("=== Quick scan results (fold 1 only, 30 epochs, EXPLORATORY) ===")
    print(df.to_string(index=False))
    print("\nNOTE: this is ONE fold -- re-run the top 1-2 combos with full 5-fold CV "
          "before reporting any of these numbers as final.")


if __name__ == "__main__":
    main()
