#!/usr/bin/env python
"""
06b_train_final_and_eval_holdout.py
======================================
Trains ONE final model on ALL non-holdout patients (no internal val split --
CV already answered "which pooling/architecture", this just fits the chosen
config on as much data as possible), for a FIXED number of epochs (no early
stopping, since there's no val signal here on purpose -- that decision was
already made via CV). Then evaluates this one model on holdout_data_for_multiclass
EXACTLY ONCE. This is the official final number for the presentation.

Run this only after 06a_train_cnn_cv.py + aggregate_fold_results.py have
confirmed mean-pooling as the choice.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                              balanced_accuracy_score, roc_auc_score, roc_curve)

from cnn_common import (CNN_MIL, CLASSES, N_CLASSES, parse_label,
                         train_one_step, predict_patient, compute_loss,
                         load_holdout_list)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--holdout_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pooling", default="mean", choices=["mean", "max", "min", "min_max"])
    p.add_argument("--unfreeze_from", default="layer4", choices=["layer4", "layer3", "all", "none"])
    p.add_argument("--instances_per_step", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30,
                   help="Should match whatever epoch count CV used, for consistency.")
    p.add_argument("--lr_head", type=float, default=1e-2)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | pooling={args.pooling} | FINAL MODEL (all non-holdout data, no val split)")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    all_patients = sorted(
        d for d in os.listdir(args.organized_dir)
        if os.path.isdir(os.path.join(args.organized_dir, d))
        and (d.startswith("cancer.") or d.startswith("normal."))
    )
    holdout_set = load_holdout_list(args.holdout_dir)
    train_ids = [p for p in all_patients if p not in holdout_set]
    holdout_ids = [p for p in all_patients if p in holdout_set]
    labels = {p: CLASSES.index(parse_label(p)) for p in all_patients}

    train_counts = np.array([sum(1 for p in train_ids if labels[p] == c) for c in range(N_CLASSES)])
    print(f"train={len(train_ids)} (all non-holdout)  holdout={len(holdout_ids)}")
    print("Train class counts:", dict(zip(CLASSES, train_counts.tolist())))

    model = CNN_MIL(unfreeze_from=args.unfreeze_from, pooling=args.pooling).to(device)
    w = torch.tensor(train_counts.sum() / (N_CLASSES * np.clip(train_counts, 1, None)),
                      dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([
            {"params": model.classifier.parameters(), "lr": args.lr_head},
            {"params": model.backbone_trainable_params(), "lr": args.lr_backbone},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([
            {"params": model.classifier.parameters(), "lr": args.lr_head},
            {"params": model.backbone_trainable_params(), "lr": args.lr_backbone},
        ], weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train_ids[:]
        rng.shuffle(shuffled)
        epoch_loss = 0.0
        for pid in shuffled:
            loss, _ = train_one_step(model, args.organized_dir, pid, labels[pid], args.image_ext,
                                      args.instances_per_step, rng, device, optimizer, criterion)
            epoch_loss += loss
        train_loss = epoch_loss / len(shuffled)
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"epoch {epoch:2d} | train_loss {train_loss:.4f}")
    # No val_loss/val_acc here on purpose -- no val split exists for the final model.

    # --- ONE-TIME holdout evaluation ---
    model.eval()
    y_true, y_score = [], []
    for pid in holdout_ids:
        scores = predict_patient(model, args.organized_dir, pid, args.image_ext, device)
        probs = torch.softmax(scores, dim=1).squeeze(0).cpu().numpy()
        y_true.append(labels[pid])
        y_score.append(probs)
    y_true = np.array(y_true)
    y_score = np.stack(y_score)
    y_pred = y_score.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    report = classification_report(y_true, y_pred, labels=range(N_CLASSES),
                                    target_names=CLASSES, digits=3, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=range(N_CLASSES),
                                         target_names=CLASSES, output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("\n==== HOLDOUT (final, one-time) ====")
    print(confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)))
    print(report)
    print("Balanced accuracy:", bal_acc)

    auc_macro = auc_weighted = None
    auc_per_class, roc_per_class = {}, {}
    try:
        auc_macro = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=range(N_CLASSES)))
        auc_weighted = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted", labels=range(N_CLASSES)))
    except ValueError as e:
        print(f"WARNING: overall AUC failed ({e})")
    for c in range(N_CLASSES):
        y_true_bin = (y_true == c).astype(int)
        if len(set(y_true_bin.tolist())) < 2:
            auc_per_class[CLASSES[c]] = None
            roc_per_class[CLASSES[c]] = None
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, c])
        roc_per_class[CLASSES[c]] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        auc_per_class[CLASSES[c]] = float(roc_auc_score(y_true_bin, y_score[:, c]))
    print("AUC macro/weighted:", auc_macro, auc_weighted)
    print("Per-class AUC:", auc_per_class)

    tag = f"final_{args.pooling}"
    torch.save({"state_dict": model.state_dict(), "pooling": args.pooling,
                "unfreeze_from": args.unfreeze_from, "classes": CLASSES},
               os.path.join(args.model_dir, f"cnn_{tag}.pt"))
    with open(os.path.join(args.model_dir, f"final_train_history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.model_dir, f"final_holdout_roc_{tag}.json"), "w") as f:
        json.dump(roc_per_class, f, indent=2)
    with open(os.path.join(args.model_dir, f"final_holdout_report_{tag}.json"), "w") as f:
        json.dump({
            "pooling": args.pooling,
            "holdout_balanced_accuracy": bal_acc,
            "holdout_confusion_matrix": cm.tolist(),
            "holdout_per_class_report": report_dict,
            "holdout_auc_macro": auc_macro, "holdout_auc_weighted": auc_weighted,
            "holdout_auc_per_class": auc_per_class,
            "class_order": CLASSES,
        }, f, indent=2)
    with open(os.path.join(args.model_dir, f"final_holdout_classification_report_{tag}.txt"), "w") as f:
        f.write(report)
    print(f"Saved final model + holdout results to {args.model_dir}")


if __name__ == "__main__":
    main()
