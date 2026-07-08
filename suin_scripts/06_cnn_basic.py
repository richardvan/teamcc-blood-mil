#!/usr/bin/env python
"""
06_cnn_finetune_mil.py  (gen2_cnn)
====================================
Server/Slurm version of MI_SVM.ipynb, upgraded from "frozen ResNet50 features +
linear SVM head" to "partially fine-tuned ResNet50 + mean-pooling MIL head".

Design decisions locked in with the user (see chat):
  - Only the last residual block (layer4 by default, see --unfreeze_from) is
    unfrozen; everything before stays frozen ImageNet weights. Full fine-tuning
    risks severe overfitting with ~150 patients total.
  - Because the backbone is no longer frozen, a full 500-image bag can't be
    forward+backward'd in one step without risking OOM. Each training step
    randomly samples --instances_per_step cell images per patient (default 32);
    a fresh random subset is drawn every time the patient is visited, so across
    epochs the model still sees the full range of a patient's cells.
  - Pooling stays mean-pooling per the user's choice (bag descriptor = mean of
    the sampled instance embeddings), matching MODE='MI' in the original notebook.
  - Loss switched from hinge loss (MultiMarginLoss) to CrossEntropyLoss: hinge
    loss made sense on top of a frozen linear-SVM head, but this is now genuine
    backprop-through-the-CNN fine-tuning, where cross-entropy is the standard
    and trains more stably. Flag if you want hinge loss back for consistency
    with the "SVM" framing.
  - Class-imbalance weighting is now ACTUALLY applied during training (the
    original notebook computed weights in a cell that ran after training
    finished, so they never took effect -- fixed here).
  - Validation/holdout inference uses ALL of a patient's cells (no sampling cap),
    since inference doesn't need to fit in a backward-pass memory budget.

This still only trains on ONE cv fold at a time (like the original notebook's
FOLD=1), because fine-tuning is expensive -- looping over all 5 folds silently
inside one job would 5x the compute cost without you asking for that. Use
--fold to pick which one; run the job again per fold if you want all 5.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from torchvision import models, transforms

CLASSES = ["control", "CBFB_MYH11", "NPM1", "PML_RARA", "RUNX1_RUNX1T1"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--organized_dir", required=True)
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--holdout_dir", default=None)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--fold", type=int, default=1, help="Which cv fold to train/validate on (1-indexed).")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--instances_per_step", type=int, default=32,
                   help="Cells randomly sampled per patient per training step (GPU memory budget).")
    p.add_argument("--unfreeze_from", default="layer4",
                   choices=["layer4", "layer3", "all", "none"],
                   help="Which ResNet50 blocks to unfreeze for fine-tuning.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr_head", type=float, default=1e-2)
    p.add_argument("--lr_backbone", type=float, default=1e-4,
                   help="Lower LR for the unfrozen backbone layers than the head.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--image_ext", default=".tif")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_hinge_loss", action="store_true",
                   help="Use MultiMarginLoss instead of CrossEntropyLoss (SVM-style, matches gen1).")
    return p.parse_args()


def resolve_defaults(args):
    project_dir = os.path.dirname(os.path.normpath(args.organized_dir))
    if args.holdout_dir is None:
        args.holdout_dir = os.path.join(project_dir, "holdout_data")
    if args.model_dir is None:
        args.model_dir = os.path.join(project_dir, "models", "gen2_cnn")
    os.makedirs(args.model_dir, exist_ok=True)
    return args


def parse_label(folder_name: str) -> str:
    parts = folder_name.split(".")
    return "control" if parts[0] == "normal" else parts[1]


def load_patient_lists(args):
    all_patients = sorted(
        d for d in os.listdir(args.organized_dir)
        if os.path.isdir(os.path.join(args.organized_dir, d))
    )
    holdout_file = os.path.join(args.holdout_dir, "holdout_patients.txt")
    holdout_set = set()
    if os.path.exists(holdout_file):
        with open(holdout_file) as f:
            holdout_set = {line.strip() for line in f if line.strip()}

    train_file = os.path.join(args.cv_dir, f"fold_{args.fold}", "train_patients.txt")
    test_file = os.path.join(args.cv_dir, f"fold_{args.fold}", "test_patients.txt")
    with open(train_file) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(test_file) as f:
        val_ids = [line.strip() for line in f if line.strip()]
    holdout_ids = [p for p in all_patients if p in holdout_set]

    return train_ids, val_ids, holdout_ids


# ------------------------------------------------------------------
# Model: partially fine-tuned ResNet50 + mean-pooling MIL head
# ------------------------------------------------------------------
class CNN_MIL(nn.Module):
    def __init__(self, unfreeze_from="layer4"):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, N_CLASSES)
        self._set_trainable(unfreeze_from)

    def _set_trainable(self, unfreeze_from):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_from == "none":
            return
        if unfreeze_from == "all":
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        blocks = {"layer3": ["layer3", "layer4"], "layer4": ["layer4"]}[unfreeze_from]
        for name, module in self.backbone.named_children():
            if name in blocks:
                for p in module.parameters():
                    p.requires_grad = True

    def backbone_trainable_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, images):
        """images: (n_cells, 3, 224, 224) for ONE patient's (sampled) cells."""
        feats = self.backbone(images)           # (n_cells, 2048)
        bag_vec = feats.mean(dim=0, keepdim=True)  # mean-pooling
        scores = self.classifier(bag_vec)        # (1, n_classes)
        return scores


PREPROCESS_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

PREPROCESS_EVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def list_patient_images(organized_dir, folder_name, image_ext):
    paths = sorted(glob.glob(os.path.join(organized_dir, folder_name, f"*{image_ext}")))
    if not paths:
        raise FileNotFoundError(f"No '{image_ext}' images for {folder_name} in {organized_dir}")
    return paths


def load_images(paths, preprocess):
    imgs = [preprocess(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs)


def train_one_step(model, organized_dir, folder_name, label, image_ext,
                    instances_per_step, rng, device, optimizer, criterion):
    paths = list_patient_images(organized_dir, folder_name, image_ext)
    if len(paths) > instances_per_step:
        paths = rng.sample(paths, instances_per_step)
    x = load_images(paths, PREPROCESS_TRAIN).to(device)
    target = torch.tensor([label], device=device)

    optimizer.zero_grad()
    scores = model(x)
    loss = criterion(scores, target)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict_patient(model, organized_dir, folder_name, image_ext, device, eval_batch=64):
    paths = list_patient_images(organized_dir, folder_name, image_ext)
    feats = []
    for i in range(0, len(paths), eval_batch):
        batch_paths = paths[i:i + eval_batch]
        x = load_images(batch_paths, PREPROCESS_EVAL).to(device)
        feats.append(model.backbone(x))
    feats = torch.cat(feats, dim=0)
    bag_vec = feats.mean(dim=0, keepdim=True)
    scores = model.classifier(bag_vec)
    return scores


def evaluate(model, organized_dir, patient_ids, labels, image_ext, device):
    model.eval()
    correct = 0
    for pid in patient_ids:
        scores = predict_patient(model, organized_dir, pid, image_ext, device)
        pred = scores.argmax(dim=1).item()
        correct += int(pred == labels[pid])
    return correct / len(patient_ids) if patient_ids else 0.0


def full_report(model, organized_dir, patient_ids, labels, image_ext, device, name):
    model.eval()
    y_true, y_pred = [], []
    for pid in patient_ids:
        scores = predict_patient(model, organized_dir, pid, image_ext, device)
        y_true.append(labels[pid])
        y_pred.append(scores.argmax(dim=1).item())
    print(f"==== {name} ====")
    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    report = classification_report(y_true, y_pred, labels=range(N_CLASSES),
                                    target_names=CLASSES, digits=3, zero_division=0)
    print(report)
    bal_acc = balanced_accuracy_score(y_true, y_pred) if y_true else 0.0
    print("Balanced accuracy:", bal_acc)
    return cm, report, bal_acc


def main():
    args = parse_args()
    args = resolve_defaults(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU -- fine-tuning a ResNet50 layer4 on CPU will be very slow. "
              "Check your sbatch --gres=gpu request if this is unexpected.")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_ids, val_ids, holdout_ids = load_patient_lists(args)
    labels = {}
    for pid in train_ids + val_ids + holdout_ids:
        labels[pid] = CLASS_TO_IDX[parse_label(pid)]

    print(f"Fold {args.fold}: train={len(train_ids)}  val={len(val_ids)}  holdout={len(holdout_ids)}")
    train_label_counts = np.array([sum(1 for p in train_ids if labels[p] == c) for c in range(N_CLASSES)])
    print("Train class counts:", dict(zip(CLASSES, train_label_counts.tolist())))
    if (train_label_counts == 0).any():
        missing = [CLASSES[i] for i in range(N_CLASSES) if train_label_counts[i] == 0]
        print(f"WARNING: these classes have ZERO training patients in fold {args.fold}: {missing}. "
              f"The model cannot learn them from this fold.")

    model = CNN_MIL(unfreeze_from=args.unfreeze_from).to(device)

    if args.use_hinge_loss:
        # weight expects a float tensor matching class order
        w = torch.tensor(train_label_counts.sum() / (N_CLASSES * np.clip(train_label_counts, 1, None)),
                          dtype=torch.float, device=device)
        criterion = nn.MultiMarginLoss(margin=1.0, weight=w)
    else:
        w = torch.tensor(train_label_counts.sum() / (N_CLASSES * np.clip(train_label_counts, 1, None)),
                          dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.SGD([
        {"params": model.classifier.parameters(), "lr": args.lr_head},
        {"params": model.backbone_trainable_params(), "lr": args.lr_backbone},
    ], weight_decay=args.weight_decay)

    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train_ids[:]
        rng.shuffle(shuffled)
        epoch_loss = 0.0
        for pid in shuffled:
            loss = train_one_step(model, args.organized_dir, pid, labels[pid], args.image_ext,
                                   args.instances_per_step, rng, device, optimizer, criterion)
            epoch_loss += loss
        val_acc = evaluate(model, args.organized_dir, val_ids, labels, args.image_ext, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d} | mean loss {epoch_loss/len(shuffled):.4f} | val acc {val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest val acc = {best_val:.3f}")

    val_cm, val_report, val_bal_acc = full_report(model, args.organized_dir, val_ids, labels,
                                                    args.image_ext, device, f"Validation (fold {args.fold})")
    hold_cm, hold_report, hold_bal_acc = full_report(model, args.organized_dir, holdout_ids, labels,
                                                       args.image_ext, device, "Holdout (final)")

    model_path = os.path.join(args.model_dir, f"cnn_mil_fold{args.fold}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "unfreeze_from": args.unfreeze_from,
        "classes": CLASSES,
        "fold": args.fold,
        "instances_per_step": args.instances_per_step,
    }, model_path)
    print("Saved model:", model_path)

    with open(os.path.join(args.model_dir, f"holdout_report_fold{args.fold}.json"), "w") as f:
        json.dump({
            "fold": args.fold,
            "best_val_balanced_accuracy": val_bal_acc,
            "holdout_balanced_accuracy": hold_bal_acc,
            "holdout_confusion_matrix": hold_cm.tolist(),
            "class_order": CLASSES,
            "unfreeze_from": args.unfreeze_from,
            "instances_per_step": args.instances_per_step,
            "use_hinge_loss": args.use_hinge_loss,
        }, f, indent=2)
    with open(os.path.join(args.model_dir, f"holdout_classification_report_fold{args.fold}.txt"), "w") as f:
        f.write(hold_report)
    print(f"Saved results to: {args.model_dir}")


if __name__ == "__main__":
    main()
