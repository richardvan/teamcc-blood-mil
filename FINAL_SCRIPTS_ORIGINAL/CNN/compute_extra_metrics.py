#!/usr/bin/env python
"""
compute_extra_metrics.py
===========================
Loads ALREADY-TRAINED checkpoints (no retraining) and re-runs inference only,
to compute precision-recall curve points and calibration (reliability) bins
for both the CV folds and the final holdout model.

Outputs (per fold, CV side):
  pr_fold{N}_{pooling}.json           per-class {precision, recall}
  calibration_fold{N}_{pooling}.json  per-class {bin_centers, observed_freq}

Outputs (final holdout side):
  pr_holdout_final_{pooling}.json
  calibration_holdout_final_{pooling}.json

Usage:
    python compute_extra_metrics.py --organized_dir ... --cv_dir ... \
        --holdout_dir ... --model_dir ... --pooling mean
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from cnn_common import (CNN_MIL, CLASSES, N_CLASSES, parse_label,
                         predict_patient, load_cv_fold_lists, load_holdout_list)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--holdout_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--n_calibration_bins", type=int, default=5,
                    help="Small holdout/val sizes need few bins -- 10 is too many for ~30 patients.")
    return p.parse_args()


def get_probs_and_labels(model, organized_dir, patient_ids, labels, image_ext, device):
    y_true, y_score = [], []
    for pid in patient_ids:
        scores = predict_patient(model, organized_dir, pid, image_ext, device)
        probs = torch.softmax(scores, dim=1).squeeze(0).cpu().numpy()
        y_true.append(labels[pid])
        y_score.append(probs)
    return np.array(y_true), np.stack(y_score)


def compute_pr_and_calibration(y_true, y_score, n_bins):
    pr_per_class, cal_per_class = {}, {}
    for c in range(N_CLASSES):
        y_true_bin = (y_true == c).astype(int)
        if len(set(y_true_bin.tolist())) < 2:
            pr_per_class[CLASSES[c]] = None
            cal_per_class[CLASSES[c]] = None
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score[:, c])
        pr_per_class[CLASSES[c]] = {"precision": precision.tolist(), "recall": recall.tolist()}
        try:
            frac_pos, mean_pred = calibration_curve(y_true_bin, y_score[:, c],
                                                       n_bins=n_bins, strategy="quantile")
            cal_per_class[CLASSES[c]] = {"mean_predicted": mean_pred.tolist(),
                                          "observed_freq": frac_pos.tolist()}
        except ValueError as e:
            print(f"WARNING: calibration failed for {CLASSES[c]} ({e})")
            cal_per_class[CLASSES[c]] = None
    return pr_per_class, cal_per_class


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

    # --- CV folds: reuse each fold's own checkpoint, evaluate on its own val_ids ---
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
        y_true, y_score = get_probs_and_labels(model, args.organized_dir, val_ids, labels,
                                                 args.image_ext, device)
        pr, cal = compute_pr_and_calibration(y_true, y_score, args.n_calibration_bins)

        with open(os.path.join(args.model_dir, f"pr_fold{fold}_{args.pooling}.json"), "w") as f:
            json.dump(pr, f, indent=2)
        with open(os.path.join(args.model_dir, f"calibration_fold{fold}_{args.pooling}.json"), "w") as f:
            json.dump(cal, f, indent=2)
        print(f"fold {fold}: saved PR + calibration ({len(val_ids)} val patients)")

    # --- Final holdout model ---
    ckpt_path = os.path.join(args.model_dir, f"cnn_final_{args.pooling}.pt")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: missing {ckpt_path}, skipping final holdout")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CNN_MIL(unfreeze_from=ckpt.get("unfreeze_from", "layer4"),
                     pooling=ckpt.get("pooling", args.pooling)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    holdout_set = load_holdout_list(args.holdout_dir)
    holdout_ids = [p for p in all_patients if p in holdout_set]
    y_true, y_score = get_probs_and_labels(model, args.organized_dir, holdout_ids, labels,
                                             args.image_ext, device)
    pr, cal = compute_pr_and_calibration(y_true, y_score, args.n_calibration_bins)

    with open(os.path.join(args.model_dir, f"pr_holdout_final_{args.pooling}.json"), "w") as f:
        json.dump(pr, f, indent=2)
    with open(os.path.join(args.model_dir, f"calibration_holdout_final_{args.pooling}.json"), "w") as f:
        json.dump(cal, f, indent=2)
    print(f"final holdout: saved PR + calibration ({len(holdout_ids)} patients)")


if __name__ == "__main__":
    main()
