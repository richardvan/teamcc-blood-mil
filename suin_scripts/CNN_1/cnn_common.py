#!/usr/bin/env python
"""
cnn_common.py
===============
Shared code for the two gen2_cnn scripts (06a_train_cnn_cv.py / 06b_train_final_and_eval_holdout.py).
Avoids duplicating the model definition and data-loading logic between them.

Label order comes from shared_functions_V2.py (team's single source of truth).
"""

import glob
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from shared_functions_V2 import SUBTYPE_TO_LABEL, LABEL_TO_SUBTYPE

CLASSES = [LABEL_TO_SUBTYPE[i] for i in range(len(SUBTYPE_TO_LABEL))]
CLASS_TO_IDX = SUBTYPE_TO_LABEL
N_CLASSES = len(CLASSES)


def parse_label(folder_name: str) -> str:
    parts = folder_name.split(".")
    return "control" if parts[0] == "normal" else parts[1]


def pool_features(feats, pooling):
    if pooling == "mean":
        return feats.mean(dim=0, keepdim=True)
    if pooling == "max":
        return feats.max(dim=0, keepdim=True).values
    if pooling == "min":
        return feats.min(dim=0, keepdim=True).values
    if pooling == "min_max":
        mn = feats.min(dim=0, keepdim=True).values
        mx = feats.max(dim=0, keepdim=True).values
        return torch.cat([mn, mx], dim=1)
    raise ValueError(f"Unknown pooling: {pooling}")


class CNN_MIL(nn.Module):
    def __init__(self, unfreeze_from="layer4", pooling="mean"):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.pooling = pooling
        in_dim = 4096 if pooling == "min_max" else 2048
        self.classifier = nn.Linear(in_dim, N_CLASSES)
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
        feats = self.backbone(images)
        bag_vec = pool_features(feats, self.pooling)
        return self.classifier(bag_vec)


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
    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])


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
    """Returns raw classifier scores (1, n_classes) using ALL of a patient's cells."""
    paths = list_patient_images(organized_dir, folder_name, image_ext)
    feats = []
    for i in range(0, len(paths), eval_batch):
        batch_paths = paths[i:i + eval_batch]
        x = load_images(batch_paths, PREPROCESS_EVAL).to(device)
        feats.append(model.backbone(x))
    feats = torch.cat(feats, dim=0)
    bag_vec = pool_features(feats, model.pooling)
    return model.classifier(bag_vec)


@torch.no_grad()
def compute_loss(model, organized_dir, patient_ids, labels, image_ext, device, criterion):
    """Mean loss over a set of patients, using ALL of each patient's cells (no epoch-to-epoch
    subsampling noise -- this is meant for tracking val/holdout loss curves, not training)."""
    if not patient_ids:
        return None
    total = 0.0
    for pid in patient_ids:
        scores = predict_patient(model, organized_dir, pid, image_ext, device)
        target = torch.tensor([labels[pid]], device=device)
        total += criterion(scores, target).item()
    return total / len(patient_ids)


def load_holdout_list(holdout_dir):
    holdout_file = os.path.join(holdout_dir, "holdout_patients.txt")
    with open(holdout_file) as f:
        return {line.strip() for line in f if line.strip()}


def load_cv_fold_lists(cv_dir, fold, all_patients):
    train_file = os.path.join(cv_dir, f"fold_{fold}", "train_patients.txt")
    test_file = os.path.join(cv_dir, f"fold_{fold}", "test_patients.txt")
    with open(train_file) as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(test_file) as f:
        val_ids = [line.strip() for line in f if line.strip()]
    return train_ids, val_ids
