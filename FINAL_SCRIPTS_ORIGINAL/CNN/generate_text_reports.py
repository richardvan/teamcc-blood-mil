#!/usr/bin/env python
"""
generate_text_reports.py
============================
Builds a single text report matching the SVM teammate's classification_report.txt
style: per-fold reports + an "Aggregate (all folds combined, OOF)" report +
(separately) the holdout report. Uses ALREADY-TRAINED checkpoints (no retraining) --
just re-runs inference to collect out-of-fold (OOF) predictions.

OOF = each of the ~156 non-holdout patients is predicted exactly once, by the
fold model that did NOT see them during training (their fold's own val set).
Pooling all 5 folds' val predictions together covers every non-holdout patient
exactly once, which is what makes the "Aggregate (all folds combined)" report
comparable to a single combined evaluation, mirroring the SVM report.

Usage:
    python generate_text_reports.py --organized_dir ... --cv_dir ... \
        --holdout_dir ... --model_dir ... --pooling mean
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, balanced_accuracy_score

from cnn_common import (CNN_MIL, CLASSES, parse_label, predict_patient,
                         load_cv_fold_lists)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--output", default=None)
    return p.parse_args()


def get_preds(model, organized_dir, patient_ids, labels, image_ext, device):
    y_true, y_pred = [], []
    for pid in patient_ids:
        scores = predict_patient(model, organized_dir, pid, image_ext, device)
        y_true.append(labels[pid])
        y_pred.append(int(scores.argmax(dim=1).item()))
    return y_true, y_pred


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_patients = sorted(
        d for d in os.listdir(args.organized_dir)
        if os.path.isdir(os.path.join(args.organized_dir, d))
        and (d.startswith("cancer.") or d.startswith("normal."))
    )
    labels = {p: CLASSES.index(parse_label(p)) for p in all_patients}

    lines = []
    oof_true, oof_pred = [], []

    for fold in range(1, 6):
        ckpt_path = os.path.join(args.model_dir, f"cnn_cv_fold{fold}_{args.pooling}.pt")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: missing {ckpt_path}, skipping fold {fold}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = CNN_MIL(unfreeze_from=ckpt.get("unfreeze_from", "layer4"),
                         pooling=ckpt.get("pooling", args.pooling)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        _, val_ids = load_cv_fold_lists(args.cv_dir, fold, all_patients)
        y_true, y_pred = get_preds(model, args.organized_dir, val_ids, labels, args.image_ext, device)
        oof_true.extend(y_true)
        oof_pred.extend(y_pred)

        report = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                        target_names=CLASSES, zero_division=0)
        lines.append(f"=== Fold {fold} ===\n{report}\n")
        print(f"fold {fold}: {len(val_ids)} val patients, balanced_acc={balanced_accuracy_score(y_true, y_pred):.3f}")

    agg_report = classification_report(oof_true, oof_pred, labels=list(range(len(CLASSES))),
                                        target_names=CLASSES, zero_division=0)
    agg_bal_acc = balanced_accuracy_score(oof_true, oof_pred)
    lines.append(f"=== Aggregate (all folds combined, OOF) ===\n{agg_report}\n")
    lines.append(f"OOF balanced accuracy: {agg_bal_acc:.4f}\n")

    out_path = args.output or os.path.join(args.model_dir, f"cv_classification_report_{args.pooling}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_path}")
    print(f"OOF balanced accuracy (all {len(oof_true)} non-holdout patients): {agg_bal_acc:.4f}")


if __name__ == "__main__":
    main()
