#!/usr/bin/env python3
"""
attention_mil_classwise_FINAL.py
=================================================
FINAL MODEL for AML genetic subtype classification: Class-wise Attention MIL
(SCEMILA-style, Hehr et al. 2023), with frozen ResNet50 instance features.

This single file consolidates everything needed to reproduce the team's final
attention-based MIL result, previously split across:
    mil_common.py, attention_mil_common.py, multi_attention_mil_common.py,
    07_6_multi_attention_mil_train.py, 07_7_multi_attention_mil_holdout_eval.py

Why class-wise attention was chosen over gated (class-shared) attention:
    Class-wise attention learns an independent attention distribution per
    class, avoiding the inter-class competition for attention weight that
    gated attention (Ilse et al., 2018) suffers from. In our experiments it
    gave better and more stable CV/holdout performance, and in particular
    recovered performance on CBFB_MYH11, which gated attention entirely
    failed to classify (F1 = 0.0) on the holdout set.

--------------------------------------------------------------------------
PREREQUISITES (unchanged from the rest of the pipeline)
--------------------------------------------------------------------------
  - metadata_for_multiclass.csv must exist directly under PROJECT_DIR,
    with columns: patient_id, folder, is_holdout, fold_1_status..fold_5_status
  - cache/cnn_features/{folder}.pt must already exist for every patient
    (run 00_extract_cnn_features.py once beforehand — frozen ResNet50
    instance embeddings, 2048-dim per cell, shared across all our MIL models)
  - Expected directory layout:
        /home/sp00001/
          shared_functions_V2.py
          blood_mil_project/
            metadata_for_multiclass.csv
            organized_data/{patient_folder}/*.tif
            cache/cnn_features/{patient_folder}.pt
            models/gen3_attention_classwise/soeun/{run_id}/...
            soeun_scripts/
              attention_mil_classwise_FINAL.py   <- this file

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  cd /home/sp00001/blood_mil_project/soeun_scripts

  # 1) Train: 5-fold CV (pick epoch count) + final model on all non-holdout patients
  python attention_mil_classwise_FINAL.py train --epochs 60

  # 2) Evaluate the trained final model on the untouched holdout set
  python attention_mil_classwise_FINAL.py eval

  # 3) (optional) Extract the single highest-attention cell per patient,
  #    for every patient in the dataset (holdout + train pool), for
  #    interpretability / qualitative review
  python attention_mil_classwise_FINAL.py rank1

Both `train` and `eval` accept --run_id (default: "latest", a symlink that
`train` updates automatically after each run), so `eval`/`rank1` always use
the most recently trained model unless a specific run_id is given.
"""

import argparse
import datetime
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc as sk_auc,
)
from sklearn.preprocessing import label_binarize


# ══════════════════════════════════════════════════════════════
# 1. PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR  = Path(__file__).resolve().parent          # .../soeun_scripts
PROJECT_DIR = SCRIPT_DIR.parent                         # .../blood_mil_project
HOME_DIR    = PROJECT_DIR.parent                        # /home/sp00001

import sys
sys.path.insert(0, str(HOME_DIR))
from shared_functions_V2 import predict_labels_and_report_performance  # noqa: E402

ORGANIZED_DIR  = PROJECT_DIR / "organized_data"
FEAT_CACHE_DIR = PROJECT_DIR / "cache" / "cnn_features"
MODEL_ROOT     = PROJECT_DIR / "models"
OUTPUT_DIR     = PROJECT_DIR

MODEL_GEN = "gen3_attention_classwise"
USER_TAG  = "soeun"
USER_TAG_DIR = MODEL_ROOT / MODEL_GEN / USER_TAG

SUBTYPE_TO_LABEL = {
    "control":       0,
    "NPM1":          1,
    "PML_RARA":      2,
    "CBFB_MYH11":    3,
    "RUNX1_RUNX1T1": 4,
}
LABEL_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_LABEL.items()}
CLASS_NAMES      = [LABEL_TO_SUBTYPE[i] for i in range(len(SUBTYPE_TO_LABEL))]
N_CLASSES        = len(CLASS_NAMES)


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════
# 2. DATA UTILITIES (metadata, bags, class weights)
# ══════════════════════════════════════════════════════════════

def parse_label_from_folder(folder_name: str) -> int:
    """'cancer.CBFB_MYH11.AOK' -> 3 ; 'normal.control.AEC' -> 0"""
    subtype = folder_name.split(".", 2)[1]
    return SUBTYPE_TO_LABEL[subtype]


def load_metadata(metadata_path: Path = None) -> pd.DataFrame:
    metadata_path = metadata_path or (PROJECT_DIR / "metadata_for_multiclass.csv")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    required = {"patient_id", "folder", "is_holdout"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata csv is missing required columns: {missing}")
    return df


def get_holdout_folders(meta_df: pd.DataFrame) -> set:
    return set(meta_df.loc[meta_df["is_holdout"] == True, "folder"])


def get_fold_split(meta_df: pd.DataFrame, fold_num: int):
    """Returns (train_folders, test_folders) for one CV fold, among non-holdout patients."""
    col = f"fold_{fold_num}_status"
    if col not in meta_df.columns:
        raise ValueError(f"metadata csv is missing column '{col}'")
    pool = meta_df.loc[meta_df["is_holdout"] == False]
    train_folders = pool.loc[pool[col] == "train", "folder"].tolist()
    test_folders  = pool.loc[pool[col] == "test",  "folder"].tolist()
    return train_folders, test_folders


def print_distribution(label: str, y: np.ndarray):
    print(f"\n  {label} ({len(y)} patients):")
    for idx in sorted(np.unique(y)):
        print(f"    {LABEL_TO_SUBTYPE[idx]:<20}: {(y == idx).sum()}")


def get_instance_filenames(folder: str):
    """.tif filenames in the same sorted order used when the features were cached."""
    return [p.name for p in sorted((ORGANIZED_DIR / folder).glob("*.tif"))]


@dataclass
class BagObject:
    patient_id: str
    instances: torch.Tensor    # (N_instances, 2048) cached ResNet50 embeddings
    true_label: int


def build_bag_objects(folders):
    bags = []
    for folder in folders:
        cache_path = FEAT_CACHE_DIR / f"{folder}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached embedding: {cache_path}\n"
                f"-> run 00_extract_cnn_features.py first."
            )
        emb = torch.load(cache_path, map_location="cpu")
        bags.append(BagObject(patient_id=folder, instances=emb,
                               true_label=parse_label_from_folder(folder)))
    return bags


def compute_class_weights(labels: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.array([max((labels == c).sum(), 1) for c in range(n_classes)], dtype=np.float64)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def relabel_bags(bags, labels):
    return [BagObject(patient_id=b.patient_id, instances=b.instances, true_label=int(l))
            for b, l in zip(bags, labels)]


# ══════════════════════════════════════════════════════════════
# 3. RUN VERSIONING (timestamped runs + a "latest" symlink)
# ══════════════════════════════════════════════════════════════

def new_run_id() -> str:
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def update_latest_symlink(base_dir: Path, run_id: str):
    base_dir.mkdir(parents=True, exist_ok=True)
    latest_link = base_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_id, target_is_directory=True)
    log(f"  latest -> {run_id}  ({latest_link})")


# ══════════════════════════════════════════════════════════════
# 4. MODEL: Class-wise Attention MIL (SCEMILA-style)
# ══════════════════════════════════════════════════════════════

class ClassWiseAttentionMIL(nn.Module):
    """
    Each of the n_classes learns its OWN attention distribution over the
    instances (rather than one attention distribution shared across all
    classes, as in gated attention / Ilse et al. 2018). This removes
    inter-class competition for attention weight.

        h_k      = encoder(x_k)                                   (per-instance encoding)
        score_ck = w_c . tanh(V_c h_k)                             (class-specific score)
        attn_:,c = softmax_k(score_:,c)                            (normalized per class)
        z_c      = sum_k attn_kc * h_k                             (class-specific bag vector)
        logit_c  = classifier(z_c)                                 (classifier shared across classes)
    """
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, attn_dim: int = 128,
                 n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        self.attn_dim = attn_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.V = nn.Linear(hidden_dim, n_classes * attn_dim, bias=False)
        self.w = nn.Parameter(torch.randn(n_classes, attn_dim) * 0.01)
        self.classifier = nn.Linear(hidden_dim, 1)   # shared across classes

    def encode_and_attend(self, bag: torch.Tensor):
        h = self.encoder(bag)                                        # (N, hidden)
        N = h.shape[0]
        v = torch.tanh(self.V(h)).view(N, self.n_classes, self.attn_dim)  # (N, C, A)
        scores = torch.einsum("nca,ca->nc", v, self.w)                # (N, C)
        attn = torch.softmax(scores, dim=0)                            # (N, C), per-class normalized
        z = torch.einsum("nc,nh->ch", attn, h)                         # (C, hidden)
        return h, z, attn

    def forward(self, bag: torch.Tensor):
        _, z, attn = self.encode_and_attend(bag)
        logits = self.classifier(z).squeeze(-1).unsqueeze(0)          # (1, n_classes)
        return logits, attn                                            # attn: (N, n_classes)


class ClassWiseAttentionMILWrapper:
    """predict_bag() interface required by shared_functions_V2."""
    def __init__(self, model: nn.Module, device: str):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def predict_bag(self, bag_instances) -> dict:
        x = bag_instances.to(self.device).float()
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        return {"pred_score": float(probs[pred_label].item()), "pred_label": pred_label}

    @torch.no_grad()
    def predict_bag_proba(self, bag_instances) -> np.ndarray:
        x = bag_instances.to(self.device).float()
        logits, _ = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_bag_with_attention(self, bag_instances) -> dict:
        """Attention weights for the PREDICTED class (i.e. 'what did the model look
        at to reach the decision it actually made')."""
        x = bag_instances.to(self.device).float()
        logits, attn = self.model(x)                # attn: (N, n_classes)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        return {
            "pred_score": float(probs[pred_label].item()),
            "pred_label": pred_label,
            "attn_weights": attn[:, pred_label].cpu().numpy(),
            "attn_weights_all_classes": attn.cpu().numpy(),
        }


# ══════════════════════════════════════════════════════════════
# 5. TRAIN / EVAL LOOP
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, bags, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for bag in bags:
        x = bag.instances.to(device).float()
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        y_true.append(bag.true_label)
        y_pred.append(int(np.argmax(probs)))
        y_proba.append(probs)
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1m, y_true, y_pred, y_proba


def train_model(model, train_bags, val_bags, device, epochs, lr, weight_decay,
                 class_weights, patience=15, verbose=True, use_early_stopping=True):
    """
    val_bags=None -> "final model" mode: train for a fixed number of epochs with
    no validation split (the epoch count was already chosen via CV).
    Returns: (model, best_val_f1_or_None, best_epoch, train_loss_history, val_loss_history)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1, best_state, best_epoch, no_improve = -1.0, None, epochs, 0
    train_loss_history, val_loss_history = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_bags))

        total_loss = 0.0
        for idx in order:
            bag = train_bags[idx]
            x = bag.instances.to(device).float()
            target = torch.tensor([bag.true_label], device=device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_bags)
        train_loss_history.append(avg_loss)

        if val_bags is not None:
            val_acc, val_f1, _, _, _ = evaluate(model, val_bags, device)

            model.eval()
            with torch.no_grad():
                vloss = sum(
                    criterion(model(b.instances.to(device).float())[0],
                              torch.tensor([b.true_label], device=device)).item()
                    for b in val_bags
                ) / len(val_bags)
            val_loss_history.append(vloss)

            if val_f1 > best_val_f1:
                best_val_f1, best_epoch, no_improve = val_f1, epoch, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                log(f"    epoch {epoch:3d}/{epochs} | train_loss {avg_loss:.4f} "
                    f"| val_loss {vloss:.4f} | val_acc {val_acc:.3f} | val_f1_macro {val_f1:.3f} "
                    f"| best_f1 {best_val_f1:.3f}")

            if use_early_stopping and no_improve >= patience:
                if verbose:
                    log(f"    early stopping @ epoch {epoch} (patience={patience})")
                break
        else:
            if verbose and (epoch % 5 == 0 or epoch == 1):
                log(f"    epoch {epoch:3d}/{epochs} | train_loss {avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_val_f1 if val_bags is not None else None), best_epoch, \
           train_loss_history, val_loss_history


def quick_eval(bags, seed, args, device, shuffle_labels=False, epochs=30):
    """Robustness check: quick 80/20 split, real vs. shuffled labels."""
    labels = np.array([b.true_label for b in bags])
    if shuffle_labels:
        labels = np.random.RandomState(seed).permutation(labels)

    idx = np.arange(len(bags))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    relabeled = relabel_bags(bags, labels)
    train_bags = [relabeled[i] for i in train_idx]
    test_bags  = [relabeled[i] for i in test_idx]

    class_weights = compute_class_weights(np.array([b.true_label for b in train_bags]))
    model = ClassWiseAttentionMIL(hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
                                   n_classes=N_CLASSES, dropout=args.dropout)
    model, _, _, _, _ = train_model(
        model, train_bags, test_bags, device, epochs=epochs, lr=args.lr,
        weight_decay=args.weight_decay, class_weights=class_weights,
        verbose=False, use_early_stopping=False,
    )
    _, _, y_true, y_pred, _ = evaluate(model, test_bags, device)
    return balanced_accuracy_score(y_true, y_pred)


# ══════════════════════════════════════════════════════════════
# 6. METRICS: AUC / ROC helpers
# ══════════════════════════════════════════════════════════════

def compute_auc_macro(y_true, y_proba, n_classes=N_CLASSES):
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                                    labels=list(range(n_classes))))
    except ValueError:
        return float("nan")


def compute_auc_per_class(y_true, y_proba, n_classes=N_CLASSES):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    out = {}
    for c in range(n_classes):
        s = y_bin[:, c].sum()
        out[c] = float("nan") if s in (0, len(y_bin)) else float(roc_auc_score(y_bin[:, c], y_proba[:, c]))
    return out


def compute_roc_curve_points(y_true, y_proba, n_classes=N_CLASSES, class_names=CLASS_NAMES):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    rows = []
    for c in range(n_classes):
        s = y_bin[:, c].sum()
        if s in (0, len(y_bin)):
            continue
        fpr, tpr, thr = roc_curve(y_bin[:, c], y_proba[:, c])
        for f, t, th in zip(fpr, tpr, thr):
            rows.append({"class": class_names[c], "fpr": f, "tpr": t, "threshold": th})
    return pd.DataFrame(rows)


def plot_multiclass_roc_grid(fold_data, class_names, n_classes, title, out_path):
    """2x3 grid: 1 summary panel + 1 panel per class. CV mode (>1 fold) shows
    faint per-fold curves + a bold mean curve (AUC mean +/- std); single-split
    mode shows one curve per class with just the AUC value."""
    is_cv = len(fold_data) > 1
    mean_fpr = np.linspace(0, 1, 100)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    summary_ax = axes.flat[0]
    class_axes = axes.flat[1:1 + n_classes]

    class_mean_tpr, class_mean_auc, class_std_auc = {}, {}, {}
    for ci, cname in enumerate(class_names):
        ax = class_axes[ci]
        tprs, aucs = [], []
        for y_true, y_proba in fold_data:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            s = y_bin[:, ci].sum()
            if s in (0, len(y_bin)):
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, ci], y_proba[:, ci])
            if is_cv:
                ax.plot(fpr, tpr, color=colors[ci], alpha=0.25, linewidth=1)
            interp = np.interp(mean_fpr, fpr, tpr); interp[0] = 0.0
            tprs.append(interp); aucs.append(sk_auc(fpr, tpr))
        if not tprs:
            class_mean_tpr[cname] = None
            ax.set_title(f"{cname} (insufficient samples)")
            continue
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc_val = float(sk_auc(mean_fpr, mean_tpr))
        std_auc_val = float(np.std(aucs)) if is_cv else 0.0
        class_mean_tpr[cname], class_mean_auc[cname], class_std_auc[cname] = mean_tpr, mean_auc_val, std_auc_val
        label = f"mean (AUC={mean_auc_val:.3f} +/- {std_auc_val:.3f})" if is_cv else f"AUC={mean_auc_val:.3f}"
        ax.plot(mean_fpr, mean_tpr, color=colors[ci], linewidth=2, label=label)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(cname); ax.legend(fontsize=8)

    for ci, cname in enumerate(class_names):
        if class_mean_tpr.get(cname) is None:
            continue
        label = (f"{cname} (AUC={class_mean_auc[cname]:.3f} +/- {class_std_auc[cname]:.3f})"
                 if is_cv else f"{cname} (AUC={class_mean_auc[cname]:.3f})")
        summary_ax.plot(mean_fpr, class_mean_tpr[cname], color=colors[ci], linewidth=2, label=label)
    summary_ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    summary_ax.set_xlabel("False Positive Rate"); summary_ax.set_ylabel("True Positive Rate")
    summary_ax.set_title("All subtypes (mean ROC)" if is_cv else "All subtypes")
    summary_ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"mean_auc_per_class": class_mean_auc, "std_auc_per_class": class_std_auc}


def plot_loss_curves_v2(fold_train_losses, fold_val_losses, title, out_path):
    """2-panel loss curve: left = train loss, right = val loss, both showing
    per-fold thin lines + a bold mean line (mean truncated to the shortest fold)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def _panel(ax, fold_losses, fold_color, mean_color, panel_title):
        fold_losses = [l for l in fold_losses if len(l) > 0]
        if not fold_losses:
            ax.set_title(f"{panel_title} (no data)")
            return
        for i, losses in enumerate(fold_losses):
            ax.plot(np.arange(1, len(losses) + 1), losses, color=fold_color, alpha=0.5,
                    linewidth=1, label="fold" if i == 0 else None)
        common_len = min(len(l) for l in fold_losses)
        mean_curve = np.mean([l[:common_len] for l in fold_losses], axis=0)
        ax.plot(np.arange(1, common_len + 1), mean_curve, color=mean_color, linewidth=2.5, label="mean")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(panel_title); ax.legend()

    _panel(axes[0], fold_train_losses, "moccasin", "darkorange", "Train loss (5 CV folds)")
    _panel(axes[1], fold_val_losses, "lightblue", "steelblue", "Val loss (5 CV folds)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 7. TRAIN: 5-fold CV + final model
# ══════════════════════════════════════════════════════════════

def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    run_id = args.run_id or new_run_id()
    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: Class-wise Attention MIL training (FINAL MODEL)")
    log(f"  Device : {device}")
    log(f"  RunID  : {run_id}")
    log(f"  SaveDir: {save_dir}")
    log("=" * 55)

    log("[1] Loading metadata...")
    meta = load_metadata()
    holdout_folders = get_holdout_folders(meta)
    all_folders = meta["folder"].tolist()
    bag_lookup = {b.patient_id: b for b in build_bag_objects(all_folders)}
    feat_dim = next(iter(bag_lookup.values())).instances.shape[1]
    train_pool_folders = [f for f in all_folders if f not in holdout_folders]
    log(f"[1] Done -- {len(all_folders)} total, {len(holdout_folders)} holdout "
        f"(excluded entirely from training), feature dim={feat_dim}")

    # ---- 5-fold CV ----
    log(f"[2] {args.n_folds}-fold CV...")
    fold_reports, fold_best_epochs, fold_aucs = [], [], []
    agg_y_true, agg_y_pred, agg_y_proba = [], [], []
    fold_data, fold_train_losses, fold_val_losses = [], [], []

    for fold in range(1, args.n_folds + 1):
        train_folders, test_folders = get_fold_split(meta, fold)
        y_tr = np.array([parse_label_from_folder(f) for f in train_folders])
        tr_idx, val_idx = train_test_split(np.arange(len(train_folders)),
                                            test_size=args.internal_val_ratio,
                                            stratify=y_tr, random_state=args.seed)
        f_train = [bag_lookup[train_folders[i]] for i in tr_idx]
        f_val   = [bag_lookup[train_folders[i]] for i in val_idx]
        f_test  = [bag_lookup[f] for f in test_folders]

        cw = compute_class_weights(np.array([b.true_label for b in f_train]))
        model = ClassWiseAttentionMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim,
                                       attn_dim=args.attn_dim, n_classes=N_CLASSES,
                                       dropout=args.dropout)
        model, _, best_epoch, tloss, vloss = train_model(
            model, f_train, f_val, device, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, class_weights=cw,
            patience=args.patience, verbose=False, use_early_stopping=True,
        )
        fold_best_epochs.append(best_epoch)
        fold_train_losses.append(tloss)
        fold_val_losses.append(vloss)

        _, _, y_true, y_pred, y_proba = evaluate(model, f_test, device)
        agg_y_true.extend(y_true.tolist()); agg_y_pred.extend(y_pred.tolist()); agg_y_proba.extend(y_proba.tolist())
        fold_data.append((y_true, y_proba))

        fauc = compute_auc_macro(y_true, y_proba)
        fold_aucs.append(fauc)
        fold_reports.append((fold, classification_report(
            y_true, y_pred, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, zero_division=0)))
        bacc = balanced_accuracy_score(y_true, y_pred)
        log(f"  Fold {fold}: n_test={len(f_test)} | balanced_acc={bacc:.3f} "
            f"| auc_macro={fauc:.3f} | best_epoch={best_epoch}")

    agg_y_true, agg_y_pred, agg_y_proba = map(np.array, (agg_y_true, agg_y_pred, agg_y_proba))
    agg_bacc = balanced_accuracy_score(agg_y_true, agg_y_pred)
    agg_f1 = f1_score(agg_y_true, agg_y_pred, average="macro", zero_division=0)
    agg_auc = compute_auc_macro(agg_y_true, agg_y_proba)
    agg_auc_pc = compute_auc_per_class(agg_y_true, agg_y_proba)
    final_epochs = int(np.median(fold_best_epochs))
    log(f"[2] CV done -- balanced_acc={agg_bacc:.3f} | f1_macro={agg_f1:.3f} | auc_macro={agg_auc:.3f}")
    log(f"[2] fold best_epochs: {fold_best_epochs} -> median {final_epochs} epochs for final model")

    report_path = save_dir / "classification_report_classwise.txt"
    with open(report_path, "w") as f:
        f.write("=== Class-wise Attention MIL hyperparameters ===\n")
        f.write(json.dumps(vars(args), indent=2) + "\n\n")
        f.write(f"fold best_epochs: {fold_best_epochs} -> final model epochs = {final_epochs}\n\n")
        for fold, rep in fold_reports:
            f.write(f"=== Fold {fold} ===\n{rep}\n")
        f.write("=== Aggregate (all folds combined) ===\n")
        f.write(classification_report(agg_y_true, agg_y_pred, labels=list(range(N_CLASSES)),
                                       target_names=CLASS_NAMES, zero_division=0))
    log(f"  classification_report -> {report_path}")

    cm_path = save_dir / "confusion_matrix_classwise.txt"
    np.savetxt(cm_path, confusion_matrix(agg_y_true, agg_y_pred, labels=list(range(N_CLASSES))), fmt="%d")

    roc_path = save_dir / "roc_curve_points_classwise.csv"
    compute_roc_curve_points(agg_y_true, agg_y_proba).to_csv(roc_path, index=False)

    roc_plot_path = save_dir / "roc_curve_plot_classwise.png"
    grid = plot_multiclass_roc_grid(fold_data, CLASS_NAMES, N_CLASSES,
        title="ROC curves -- Class-wise Attention MIL (CV, per-fold + mean per subtype)",
        out_path=roc_plot_path)

    auc_path = save_dir / "auc_metrics_classwise.json"
    auc_path.write_text(json.dumps({
        "per_fold_auc_macro": fold_aucs,
        "aggregate_auc_macro_pooled": agg_auc,
        "aggregate_auc_per_class_pooled": {CLASS_NAMES[c]: v for c, v in agg_auc_pc.items()},
        "mean_auc_per_class_over_folds": grid["mean_auc_per_class"],
        "std_auc_per_class_over_folds": grid["std_auc_per_class"],
        "fold_best_epochs": fold_best_epochs,
        "final_model_epochs": final_epochs,
    }, indent=2))
    log(f"  ROC plot / AUC metrics saved -> {save_dir}")

    # ---- Final model: all non-holdout patients, no val split, fixed epoch count ----
    log(f"[3] Training final model on all {len(train_pool_folders)} non-holdout patients "
        f"(no val split, fixed {final_epochs} epochs)...")
    train_pool_bags = [bag_lookup[f] for f in train_pool_folders]
    print_distribution("Final train set (full, no val split)",
                        np.array([b.true_label for b in train_pool_bags]))

    cw = compute_class_weights(np.array([b.true_label for b in train_pool_bags]))
    model = ClassWiseAttentionMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim,
                                   attn_dim=args.attn_dim, n_classes=N_CLASSES, dropout=args.dropout)
    model, _, _, final_train_loss, _ = train_model(
        model, train_pool_bags, None, device, epochs=final_epochs, lr=args.lr,
        weight_decay=args.weight_decay, class_weights=cw, verbose=True, use_early_stopping=False,
    )
    log(f"[3] Done (fixed {final_epochs} epochs, no val)")

    loss_plot_path = save_dir / "loss_curve_plot_classwise.png"
    plot_loss_curves_v2(fold_train_losses, fold_val_losses,
        title=f"{MODEL_GEN} loss curves (class-wise attention)", out_path=loss_plot_path)

    model_dir = USER_TAG_DIR / run_id
    model_path = model_dir / f"{args.model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": feat_dim, "hidden_dim": args.hidden_dim, "attn_dim": args.attn_dim,
        "n_classes": N_CLASSES, "dropout": args.dropout, "classes": CLASS_NAMES,
        "final_epochs": final_epochs,
    }, model_path)
    log(f"  model saved -> {model_path}")

    # ---- Robustness check: real vs. shuffled labels ----
    log(f"[4] Robustness check ({args.n_robustness_seeds} seeds, real vs shuffled labels)...")
    real_scores = [quick_eval(train_pool_bags, s, args, device, shuffle_labels=False)
                   for s in range(args.n_robustness_seeds)]
    shuf_scores = [quick_eval(train_pool_bags, s, args, device, shuffle_labels=True)
                   for s in range(args.n_robustness_seeds)]
    log(f"  real     : {np.round(real_scores, 3).tolist()} mean {np.mean(real_scores):.3f}")
    log(f"  shuffled : {np.round(shuf_scores, 3).tolist()} mean {np.mean(shuf_scores):.3f}")
    if np.mean(shuf_scores) > 0.35:
        log("  [WARN] shuffled-label accuracy is suspiciously high -- check for data leakage")
    (save_dir / "robustness_check_classwise.json").write_text(json.dumps({
        "real_scores": real_scores, "real_mean": float(np.mean(real_scores)),
        "shuffled_scores": shuf_scores, "shuffled_mean": float(np.mean(shuf_scores)),
    }, indent=2))

    # ---- PCA of class-specific bag vectors (concatenated across classes) ----
    log("[5] PCA visualization...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    wrapper = ClassWiseAttentionMILWrapper(model, device)
    groups, vecs, labels = [], [], []
    for folder in train_pool_folders:
        bag = bag_lookup[folder]
        with torch.no_grad():
            _, z, _ = model.encode_and_attend(bag.instances.to(device).float())
        groups.append(folder); vecs.append(z.cpu().numpy().flatten()); labels.append(bag.true_label)
    X = np.array(vecs); groups = np.array(groups); labels = np.array(labels)
    np.save(save_dir / "X_bagvec_classwise.npy", X)
    np.save(save_dir / "groups_bagvec_classwise.npy", groups)
    np.save(save_dir / "labels_bagvec_classwise.npy", labels)

    Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
    plt.figure()
    for ci, cname in enumerate(CLASS_NAMES):
        m = labels == ci
        plt.scatter(Z[m, 0], Z[m, 1], label=cname, alpha=.6, c=[colors[ci]])
    plt.legend(); plt.title("Patient bag embeddings (class-wise attention, PCA)")
    pca_plot_path = save_dir / "pca_plot_classwise.png"
    plt.savefig(pca_plot_path, bbox_inches="tight"); plt.close()

    update_latest_symlink(USER_TAG_DIR, run_id)
    log("=" * 55)
    log("DONE: training complete")
    log(f"  model              : {model_path}")
    log(f"  final_model_epochs : {final_epochs}")
    log(f"  classification_report: {report_path}")
    log(f"  confusion_matrix   : {cm_path}")
    log(f"  auc_metrics        : {auc_path}")
    log(f"  roc_curve_plot     : {roc_plot_path}")
    log(f"  loss_curve_plot    : {loss_plot_path}")
    log(f"  pca_plot           : {pca_plot_path}")
    log("=" * 55)


# ══════════════════════════════════════════════════════════════
# 8. EVAL: holdout evaluation of the trained final model
# ══════════════════════════════════════════════════════════════

def cmd_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = args.run_id
    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: Class-wise Attention MIL holdout evaluation")
    log(f"  Device : {device}")
    log("=" * 55)

    model_path = USER_TAG_DIR / run_id / f"{args.model_name}.pt"
    log(f"[1] Loading model -> {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = ClassWiseAttentionMIL(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"],
                                   attn_dim=ckpt["attn_dim"], n_classes=ckpt["n_classes"],
                                   dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    wrapper = ClassWiseAttentionMILWrapper(model, device)
    log("[1] Done")

    log("[2] Loading metadata and holdout bags...")
    meta = load_metadata()
    holdout_folders = sorted(get_holdout_folders(meta))
    holdout_bags = build_bag_objects(holdout_folders)
    log(f"[2] Done -- {len(holdout_bags)} holdout patients")

    log("[3] Predicting on holdout...")
    y_true = np.array([b.true_label for b in holdout_bags])
    y_pred = np.array([wrapper.predict_bag(b.instances)["pred_label"] for b in holdout_bags])
    y_proba = np.array([wrapper.predict_bag_proba(b.instances) for b in holdout_bags])

    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc_macro = compute_auc_macro(y_true, y_proba)
    auc_pc = compute_auc_per_class(y_true, y_proba)
    log(f"[3] holdout balanced accuracy: {bacc:.3f}")
    log(f"[3] holdout F1 (macro)       : {f1m:.3f}")
    log(f"[3] holdout AUC (macro, OvR) : {auc_macro:.3f}")
    for c, name in enumerate(CLASS_NAMES):
        log(f"       AUC[{name:<16}]: {auc_pc[c]:.3f}")

    report = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                    target_names=CLASS_NAMES, zero_division=0)
    print(report, flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print(cm, flush=True)

    (save_dir / "holdout_classification_report_classwise.txt").write_text(report)
    np.savetxt(save_dir / "holdout_confusion_matrix_classwise.txt", cm, fmt="%d")

    roc_df = compute_roc_curve_points(y_true, y_proba)
    roc_df.to_csv(save_dir / "holdout_roc_curve_points_classwise.csv", index=False)
    roc_plot_path = save_dir / "holdout_roc_curve_plot_classwise.png"
    grid = plot_multiclass_roc_grid([(y_true, y_proba)], CLASS_NAMES, N_CLASSES,
        title="ROC curves -- Class-wise Attention MIL (holdout)", out_path=roc_plot_path)
    (save_dir / "holdout_auc_metrics_classwise.json").write_text(json.dumps({
        "auc_macro": auc_macro,
        "auc_per_class": {CLASS_NAMES[c]: v for c, v in auc_pc.items()},
        "auc_per_class_grid": grid["mean_auc_per_class"],
    }, indent=2))

    log("[4] Saving standard team-format outputs via shared_functions_V2...")
    bag_df, metrics_df = predict_labels_and_report_performance(
        model=wrapper, holdout_bags=holdout_bags, model_gen=MODEL_GEN,
        model_name=args.model_name, output_dir=str(OUTPUT_DIR / USER_TAG / run_id),
    )
    print(metrics_df.to_string(index=False), flush=True)

    log("[5] Extracting top-attention cells (attention weights for the predicted class)...")
    rows = []
    for bag in holdout_bags:
        result = wrapper.predict_bag_with_attention(bag.instances)
        attn = result["attn_weights"]
        filenames = get_instance_filenames(bag.patient_id)
        if len(filenames) != len(attn):
            continue
        top_idx = np.argsort(-attn)[:args.top_k]
        for rank, i in enumerate(top_idx, start=1):
            rows.append({
                "patient_id": bag.patient_id, "true_label": CLASS_NAMES[bag.true_label],
                "pred_label": CLASS_NAMES[result["pred_label"]], "rank": rank,
                "filename": filenames[i], "attn_weight": float(attn[i]),
            })
    pd.DataFrame(rows).to_csv(save_dir / "attention_top_cells_holdout_classwise.csv", index=False)

    log("=" * 55)
    log("DONE: holdout evaluation complete")
    log(f"  balanced_accuracy : {bacc:.3f}")
    log(f"  f1_macro          : {f1m:.3f}")
    log(f"  auc_macro         : {auc_macro:.3f}")
    log("=" * 55)


# ══════════════════════════════════════════════════════════════
# 9. (optional) RANK-1: single highest-attention cell per patient, ALL patients
# ══════════════════════════════════════════════════════════════

def cmd_rank1(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = args.run_id
    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = USER_TAG_DIR / run_id / f"{args.model_name}.pt"
    ckpt = torch.load(model_path, map_location=device)
    model = ClassWiseAttentionMIL(in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"],
                                   attn_dim=ckpt["attn_dim"], n_classes=ckpt["n_classes"],
                                   dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    wrapper = ClassWiseAttentionMILWrapper(model, device)

    meta = load_metadata()
    holdout_folders = get_holdout_folders(meta)
    all_folders = meta["folder"].tolist()
    bags = build_bag_objects(all_folders)

    rows = []
    for bag in bags:
        result = wrapper.predict_bag_with_attention(bag.instances)
        attn = result["attn_weights"]
        filenames = get_instance_filenames(bag.patient_id)
        if len(filenames) != len(attn):
            continue
        top_i = int(np.argmax(attn))
        rows.append({
            "patient_id": bag.patient_id,
            "split": "holdout" if bag.patient_id in holdout_folders else "train_pool",
            "true_subtype": CLASS_NAMES[bag.true_label],
            "pred_subtype": CLASS_NAMES[result["pred_label"]],
            "correct": int(bag.true_label == result["pred_label"]),
            "n_instances": len(attn), "rank": 1,
            "filename": filenames[top_i], "attn_weight": float(attn[top_i]),
        })
    out_path = save_dir / "attention_rank1_all_patients.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log(f"Saved -> {out_path}")


# ══════════════════════════════════════════════════════════════
# 10. CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Class-wise Attention MIL -- FINAL model (train / eval / rank1)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--hidden_dim", type=int, default=256)
        p.add_argument("--attn_dim", type=int, default=128)
        p.add_argument("--dropout", type=float, default=0.3)
        p.add_argument("--model_name", type=str, default="multi_attention_mil_v1")

    p_train = sub.add_parser("train", help="5-fold CV + final model training")
    add_common(p_train)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=60,
                          help="max epochs per CV fold (early stopping applies)")
    p_train.add_argument("--patience", type=int, default=15)
    p_train.add_argument("--internal_val_ratio", type=float, default=0.15)
    p_train.add_argument("--n_folds", type=int, default=5)
    p_train.add_argument("--n_robustness_seeds", type=int, default=10)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--run_id", type=str, default=None,
                          help="defaults to a new timestamp; 'latest' symlink is updated automatically")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate the trained final model on the holdout set")
    add_common(p_eval)
    p_eval.add_argument("--top_k", type=int, default=10)
    p_eval.add_argument("--run_id", type=str, default="latest")
    p_eval.set_defaults(func=cmd_eval)

    p_rank1 = sub.add_parser("rank1", help="Extract the single top-attention cell for every patient")
    add_common(p_rank1)
    p_rank1.add_argument("--run_id", type=str, default="latest")
    p_rank1.set_defaults(func=cmd_rank1)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
