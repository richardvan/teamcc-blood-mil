#!/usr/bin/env python3
"""
CNN-MIL: AML Subtype Classification (5-class)
방법 A: fold_1 기반 단일 학습
=====================================

분할 방식:
  - holdout_patients.txt → test (28명, 최종 평가, 학습 미사용)
  - cv_splits/fold_1/train_patients.txt → train (121명)
  - cv_splits/fold_1/test_patients.txt  → validation (30명, early stopping 기준)

파일 위치:
  blood_mil_project/
    scripts/shared_functions.py
    organized_data/
    holdout_data/holdout_patients.txt
    cv_splits/fold_1/train_patients.txt
    cv_splits/fold_1/test_patients.txt
    models/gen2_cnn/cnn_mil_fold1_v1.pt
    soeun_scripts/06a_cnn_mil_fold1.py  ← 이 스크립트

Usage:
  python /home/sp00001/blood_mil_project/soeun_scripts/06a_cnn_mil_fold1.py
  python /home/sp00001/blood_mil_project/soeun_scripts/06a_cnn_mil_fold1.py --epochs 50 --lr 5e-5
"""

import sys
import json
import time
import datetime
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    confusion_matrix, classification_report,
)

# ── 경로 자동 설정 ────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
from shared_functions import compute_foreground_mask, stretch_to_unit_range

# ── 레이블 정의 ───────────────────────────────────────────────
SUBTYPE_TO_LABEL = {
    "control": 0, "NPM1": 1, "PML_RARA": 2,
    "CBFB_MYH11": 3, "RUNX1_RUNX1T1": 4,
}
LABEL_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_LABEL.items()}
CLASS_NAMES      = [LABEL_TO_SUBTYPE[i] for i in range(len(SUBTYPE_TO_LABEL))]

SCEMILA_F1 = {
    "PML_RARA": (0.86, 0.05), "NPM1": (0.75, 0.06),
    "CBFB_MYH11": (0.69, 0.09), "RUNX1_RUNX1T1": (0.75, 0.15),
}


# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_label(folder_name: str) -> int:
    subtype = folder_name.split(".", 2)[1]
    if subtype not in SUBTYPE_TO_LABEL:
        raise ValueError(f"Unknown subtype: {subtype}")
    return SUBTYPE_TO_LABEL[subtype]


def load_txt(path: Path) -> list:
    return [l.strip() for l in open(path) if l.strip()]


# ──────────────────────────────────────────────────────────────
# 팀 표준 전처리
# ──────────────────────────────────────────────────────────────

def preprocess_image_for_cnn(path: str) -> np.ndarray:
    img  = Image.open(path).convert("RGB")
    arr  = np.asarray(img, dtype=np.float32) / 255.0
    mask = compute_foreground_mask(arr)
    arr_norm = np.empty_like(arr)
    for ch in range(3):
        arr_norm[:, :, ch] = stretch_to_unit_range(arr[:, :, ch], mask)
    return arr_norm   # (H, W, 3) float32


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class CellDataset(Dataset):
    def __init__(self, patient_dir: Path, augment: bool = False,
                 max_cells: int = 0, seed: int = 42):
        tifs = sorted(patient_dir.glob("*.tif"))
        if not tifs:
            raise ValueError(f"No .tif in {patient_dir}")
        if max_cells and len(tifs) > max_cells:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(tifs), size=max_cells, replace=False)
            tifs = [tifs[i] for i in sorted(idx)]
        self.tifs    = tifs
        self.resize  = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if augment else None

    def __len__(self): return len(self.tifs)

    def __getitem__(self, idx):
        arr    = preprocess_image_for_cnn(str(self.tifs[idx]))
        img    = Image.fromarray((arr * 255).astype(np.uint8))
        tensor = self.resize(self.to_tensor(img))
        if self.augment:
            tensor = self.augment(tensor)
        return tensor


# ──────────────────────────────────────────────────────────────
# 모델: ResNet34 + Mean Pooling + FC
# ──────────────────────────────────────────────────────────────

class CNNMILModel(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        feats    = self.feature_extractor(bag).squeeze(-1).squeeze(-1)  # (N,512)
        bag_feat = feats.mean(dim=0)                                     # (512,)
        return self.classifier(bag_feat)                                  # (5,)


# ──────────────────────────────────────────────────────────────
# 환자 1명 추론
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_patient(model, patient_dir, device, max_cells=0, batch_size=32):
    ds     = CellDataset(patient_dir, augment=False, max_cells=max_cells)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_feats = []
    for batch in loader:
        f = model.feature_extractor(batch.to(device)).squeeze(-1).squeeze(-1)
        all_feats.append(f.cpu())
    feats  = torch.cat(all_feats).mean(dim=0)
    logits = model.classifier(feats)
    return {"pred_label": int(logits.argmax()), "pred_score": float(logits.max())}


# ──────────────────────────────────────────────────────────────
# 1 epoch 학습
# ──────────────────────────────────────────────────────────────

def train_epoch(model, folders, organized_dir, device,
                optimizer, criterion, max_cells, batch_size):
    model.train()
    total_loss, correct = 0.0, 0
    for folder in folders:
        label  = parse_label(folder)
        ds     = CellDataset(organized_dir / folder, augment=True, max_cells=max_cells)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
        all_feats = []
        for batch in loader:
            f = model.feature_extractor(batch.to(device)).squeeze(-1).squeeze(-1)
            all_feats.append(f)
        feats  = torch.cat(all_feats).mean(dim=0)
        logits = model.classifier(feats.unsqueeze(0))
        target = torch.tensor([label], device=device)
        loss   = criterion(logits, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        correct    += int(logits.argmax(1).item() == label)
    n = len(folders)
    return total_loss / n, correct / n


# ──────────────────────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────────────────────

def evaluate(model, folders, organized_dir, device, max_cells=0, batch_size=32):
    y_true, y_pred = [], []
    for folder in folders:
        r = predict_patient(model, organized_dir / folder, device, max_cells, batch_size)
        y_true.append(parse_label(folder))
        y_pred.append(r["pred_label"])
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            y_true, y_pred)


# ──────────────────────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────────────────────

def report_and_save(y_true, y_pred, pred_scores, patient_ids,
                    output_dir, model_name):
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_cls  = f1_score(y_true, y_pred,
                           labels=list(range(len(CLASS_NAMES))),
                           average=None, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(y_true, y_pred,
                                   labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, zero_division=0)

    print(f"\n{'='*60}", flush=True)
    print(f"Holdout 평가 결과 — {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Accuracy     : {acc:.3f}", flush=True)
    print(f"  F1 (macro)   : {f1_macro:.3f}", flush=True)
    print(f"  F1 (weighted): {f1_weighted:.3f}", flush=True)
    print(f"\n  클래스별 F1:", flush=True)
    for cls, f1 in zip(CLASS_NAMES, f1_per_cls):
        print(f"    {cls:<20}: {f1:.3f}", flush=True)
    print(f"\n  Confusion matrix (행=정답, 열=예측):", flush=True)
    header = "            " + "  ".join(f"{n[:6]:>6}" for n in CLASS_NAMES)
    print(header, flush=True)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:10]:<12}" +
              "  ".join(f"{v:>6}" for v in row), flush=True)
    print(f"\n  Classification report:\n{report}", flush=True)
    print(f"\n  논문(SCEMILA) vs {model_name}:", flush=True)
    print(f"  {'클래스':<20} {'SCEMILA':>12} {model_name:>12}", flush=True)
    print("  " + "-" * 48, flush=True)
    for cls in ["PML_RARA", "NPM1", "CBFB_MYH11", "RUNX1_RUNX1T1"]:
        s_m, s_s = SCEMILA_F1[cls]
        val      = f1_per_cls[SUBTYPE_TO_LABEL[cls]]
        print(f"  {cls:<20} {s_m:.2f}±{s_s:.2f}    {val:.3f}", flush=True)

    tag      = f"gen2_cnn_{model_name}"
    pred_dir = output_dir / "predictions"
    perf_dir = output_dir / "performance"
    pred_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "patient_id":   patient_ids,
        "true_label":   y_true,   "pred_label":   y_pred,
        "pred_score":   pred_scores,
        "true_subtype": [LABEL_TO_SUBTYPE[l] for l in y_true],
        "pred_subtype": [LABEL_TO_SUBTYPE[l] for l in y_pred],
        "correct":      (np.array(y_true) == np.array(y_pred)).astype(int),
        "model_gen": "gen2_cnn", "model_name": model_name,
    }).to_csv(pred_dir / f"predicted_labels.{tag}.csv", index=False)

    perf_row = {
        "model_gen": "gen2_cnn", "model_name": model_name,
        "n_holdout": len(y_true),
        "accuracy":        round(float(acc),         4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro":    round(float(f1_macro),    4),
        "weighted_f1": round(float(f1_weighted), 4),
    }
    for cls in CLASS_NAMES:
        perf_row[f"f1_{cls}"] = round(float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]), 4)
    pd.DataFrame([perf_row]).to_csv(
        perf_dir / f"performance_metrics.{tag}.csv", index=False)
    (perf_dir / f"classification_report.{tag}.txt").write_text(report)
    (perf_dir / f"performance_metrics_5class.{tag}.json").write_text(
        json.dumps({
            "model_gen": "gen2_cnn", "model_name": model_name,
            "accuracy": float(acc), "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]])
                             for cls in CLASS_NAMES},
        }, indent=2))


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN-MIL 방법A: fold_1 기반 단일 학습")
    parser.add_argument("--fold",       type=int,   default=1)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--max_cells",  type=int,   default=200)
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_dir   = PROJECT_DIR
    organized_dir = project_dir / "organized_data"
    holdout_dir   = project_dir / "holdout_data"
    cv_dir        = project_dir / "cv_splits"
    model_name    = f"cnn_mil_fold{args.fold}_v1"

    log("=" * 55)
    log(f"START: CNN-MIL 방법A — fold_{args.fold} 기반")
    log(f"  Device    : {device}")
    log(f"  Epochs    : {args.epochs}  LR: {args.lr}")
    log(f"  Max cells : {args.max_cells or 'all'}")
    log("=" * 55)

    # ── Step 1: 분할 로드 ─────────────────────────────────────
    log(f"[Step 1/5] cv_splits/fold_{args.fold} 로드...")
    train_folders   = load_txt(cv_dir / f"fold_{args.fold}" / "train_patients.txt")
    val_folders     = load_txt(cv_dir / f"fold_{args.fold}" / "test_patients.txt")
    holdout_folders = load_txt(holdout_dir / "holdout_patients.txt")

    log(f"[Step 1/5] 완료")
    log(f"  Train      : {len(train_folders)}명")
    log(f"  Validation : {len(val_folders)}명  ← fold_{args.fold} test")
    log(f"  Holdout    : {len(holdout_folders)}명  ← 최종 평가")

    # 분포 출력
    for split_name, folders in [("Train", train_folders),
                                  ("Val",   val_folders),
                                  ("Holdout", holdout_folders)]:
        cnt = defaultdict(int)
        for f in folders: cnt[LABEL_TO_SUBTYPE[parse_label(f)]] += 1
        print(f"\n  {split_name} ({len(folders)}명):", flush=True)
        for cls in CLASS_NAMES:
            print(f"    {cls:<20}: {cnt.get(cls,0)}명", flush=True)

    # ── Step 2: 모델 초기화 ───────────────────────────────────
    log("\n[Step 2/5] ResNet34 초기화...")
    model     = CNNMILModel(num_classes=len(CLASS_NAMES)).to(device)
    # 클래스 불균형 보정 가중치
    label_cnt = defaultdict(int)
    for f in train_folders: label_cnt[parse_label(f)] += 1
    weights   = torch.tensor([
        len(train_folders) / (len(CLASS_NAMES) * label_cnt.get(i, 1))
        for i in range(len(CLASS_NAMES))
    ], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    log("[Step 2/5] 완료")

    # ── Step 3: 학습 ──────────────────────────────────────────
    log("[Step 3/5] 학습 시작...")
    best_val_f1, patience_cnt, best_state = -1.0, 0, None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_folders, organized_dir, device,
            optimizer, criterion, args.max_cells, args.batch_size)
        val_acc, val_f1, _, _ = evaluate(
            model, val_folders, organized_dir, device,
            args.max_cells, args.batch_size)
        scheduler.step()

        log(f"  Epoch {epoch:02d}/{args.epochs} | "
            f"loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
            f"val_acc={val_acc:.3f} val_f1={val_f1:.3f} | "
            f"{time.time()-t0:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            log(f"    ★ Best val F1: {best_val_f1:.3f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log(f"  Early stopping (patience={args.patience})")
                break

    log(f"[Step 3/5] 완료 — Best val F1: {best_val_f1:.3f}")

    # ── Step 4: 모델 저장 ─────────────────────────────────────
    log("[Step 4/5] 모델 저장...")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model_dir = project_dir / "models" / "gen2_cnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    log(f"[Step 4/5] 완료 → {model_path}")

    # ── Step 5: holdout 평가 ──────────────────────────────────
    log("[Step 5/5] holdout 평가 중...")
    y_true, y_pred, pred_scores, patient_ids = [], [], [], []
    for i, folder in enumerate(holdout_folders, 1):
        r = predict_patient(model, organized_dir / folder,
                            device, args.max_cells, args.batch_size)
        tl = parse_label(folder)
        y_true.append(tl); y_pred.append(r["pred_label"])
        pred_scores.append(r["pred_score"]); patient_ids.append(folder)
        log(f"  [{i}/{len(holdout_folders)}] {folder} → "
            f"pred={LABEL_TO_SUBTYPE[r['pred_label']]} "
            f"(true={LABEL_TO_SUBTYPE[tl]})")

    log("[Step 5/5] 완료")
    report_and_save(y_true, y_pred, pred_scores, patient_ids,
                    project_dir, model_name)

    log("=" * 55)
    log("DONE")
    log(f"  model : {model_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
