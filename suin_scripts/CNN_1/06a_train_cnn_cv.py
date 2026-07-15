#!/usr/bin/env python
"""
06a_train_cnn_cv.py
======================
CV-only training script: trains ONE fold at a time (--fold 1..5), evaluates on
that fold's own test partition (val_ids). Does NOT touch holdout data at all --
holdout evaluation now lives in 06b_train_final_and_eval_holdout.py, run once
after CV has picked mean-pooling as the final choice.

Saves per fold:
  - cnn_cv_fold{N}.pt                  model checkpoint
  - cv_history_fold{N}.json            per-epoch {train_loss, val_loss, val_acc}
  - cv_roc_fold{N}.json                per-class {fpr, tpr} for the best-epoch model on val_ids
  - cv_report_fold{N}.json             val confusion matrix, per-class report, AUC (macro/weighted/per-class)
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
                         load_cv_fold_lists)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--pooling", default="mean", choices=["mean", "max", "min", "min_max"])
    p.add_argument("--unfreeze_from", default="layer4", choices=["layer4", "layer3", "all", "none"])
    p.add_argument("--instances_per_step", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr_head", type=float, default=1e-2)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_probs_and_labels(model, organized_dir, patient_ids, labels, image_ext, device):
    y_true, y_score = [], []
    for pid in patient_ids:
        scores = predict_patient(model, organized_dir, pid, image_ext, device)
        probs = torch.softmax(scores, dim=1).squeeze(0).cpu().numpy()
        y_true.append(labels[pid])
        y_score.append(probs)
    return np.array(y_true), np.stack(y_score)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | pooling={args.pooling} | fold={args.fold}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    all_patients = sorted(
        d for d in os.listdir(args.organized_dir)
        if os.path.isdir(os.path.join(args.organized_dir, d))
        and (d.startswith("cancer.") or d.startswith("normal."))
    )
    train_ids, val_ids = load_cv_fold_lists(args.cv_dir, args.fold, all_patients)
    labels = {p: CLASSES.index(parse_label(p)) for p in all_patients}

    train_counts = np.array([sum(1 for p in train_ids if labels[p] == c) for c in range(N_CLASSES)])
    print(f"train={len(train_ids)} val={len(val_ids)}")
    print("Train class counts:", dict(zip(CLASSES, train_counts.tolist())))
    if (train_counts == 0).any():
        missing = [CLASSES[i] for i in range(N_CLASSES) if train_counts[i] == 0]
        print(f"WARNING: zero training patients for: {missing}")

    model = CNN_MIL(unfreeze_from=args.unfreeze_from, pooling=args.pooling).to(device)
    w = torch.tensor(train_counts.sum() / (N_CLASSES * np.clip(train_counts, 1, None)),
                      dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.SGD([
        {"params": model.classifier.parameters(), "lr": args.lr_head},
        {"params": model.backbone_trainable_params(), "lr": args.lr_backbone},
    ], weight_decay=args.weight_decay)

    best_val_acc, best_state = 0.0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train_ids[:]
        rng.shuffle(shuffled)
        epoch_loss = 0.0
        for pid in shuffled:
            loss = train_one_step(model, args.organized_dir, pid, labels[pid], args.image_ext,
                                   args.instances_per_step, rng, device, optimizer, criterion)
            epoch_loss += loss
        train_loss = epoch_loss / len(shuffled)

        model.eval()
        val_loss = compute_loss(model, args.organized_dir, val_ids, labels, args.image_ext, device, criterion)
        correct = 0
        for pid in val_ids:
            scores = predict_patient(model, args.organized_dir, pid, args.image_ext, device)
            correct += int(scores.argmax(dim=1).item() == labels[pid])
        val_acc = correct / len(val_ids) if val_ids else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final val evaluation with the best-epoch model ---
    y_true, y_score = get_probs_and_labels(model, args.organized_dir, val_ids, labels, args.image_ext, device)
    y_pred = y_score.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    report_dict = classification_report(y_true, y_pred, labels=range(N_CLASSES),
                                         target_names=CLASSES, output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

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

    print(f"Val balanced_accuracy={bal_acc:.3f}  AUC_macro={auc_macro}  AUC_weighted={auc_weighted}")

    tag = f"fold{args.fold}_{args.pooling}"
    torch.save({"state_dict": model.state_dict(), "pooling": args.pooling,
                "unfreeze_from": args.unfreeze_from, "classes": CLASSES, "fold": args.fold},
               os.path.join(args.model_dir, f"cnn_cv_{tag}.pt"))

    with open(os.path.join(args.model_dir, f"cv_history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.model_dir, f"cv_roc_{tag}.json"), "w") as f:
        json.dump(roc_per_class, f, indent=2)
    with open(os.path.join(args.model_dir, f"cv_report_{tag}.json"), "w") as f:
        json.dump({
            "fold": args.fold, "pooling": args.pooling,
            "val_balanced_accuracy": bal_acc,
            "val_confusion_matrix": cm.tolist(),
            "val_per_class_report": report_dict,
            "val_auc_macro": auc_macro, "val_auc_weighted": auc_weighted,
            "val_auc_per_class": auc_per_class,
            "class_order": CLASSES,
        }, f, indent=2)
    print(f"Saved fold {args.fold} results to {args.model_dir}")


if __name__ == "__main__":
    main()
