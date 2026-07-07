#!/usr/bin/env python3
"""
CNN-MIL: AML Subtype Classification (5-class) — holdout 기반, GPU 학습

SVM-MIL과의 차이:
  - 인스턴스 피처 = 손수 설계한 GLCM/컬러 대신 frozen ResNet 임베딩
    (00_extract_cnn_features.py 캐시 사용)
  - Bag pooling(mean 또는 max) 뒤에 얕은 MLP를 붙여 end-to-end로 학습
    (SVM은 pooled 벡터에 대해 linear kernel/RBF만 사용했다면,
     여기서는 pooling+classifier를 함께 gradient descent로 최적화)

전제 조건:
  python 00_extract_cnn_features.py 를 먼저 실행해 cache/cnn_features/*.pt 준비

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 06_1_cnn_mil_subtype_holdout.py --pooling mean --epochs 60
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
)

from mil_common import (
    PROJECT_DIR, MODEL_ROOT, OUTPUT_DIR, CLASS_NAMES, N_CLASSES,
    LABEL_TO_SUBTYPE, SUBTYPE_TO_LABEL,
    log, load_holdout_folders, list_patient_dirs, print_distribution,
    BagObject, build_bag_objects, compute_class_weights, parse_label_from_folder,
    ORGANIZED_DIR,
)

from shared_functions_V1 import predict_labels_and_report_performance


# ──────────────────────────────────────────────────────────────
# 1. 모델 정의
# ──────────────────────────────────────────────────────────────

class CNNMIL(nn.Module):
    """
    pooling='mean' | 'max'  : 인스턴스 임베딩을 먼저 pooling한 뒤 분류
                              (embedding-level pooling, 'MI' 스타일)
    pooling='instance_max'  : 인스턴스별로 먼저 분류하고 최댓값을 취함
                              (instance-level pooling, 'mi' 스타일)
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 pooling: str = "mean", dropout: float = 0.3):
        super().__init__()
        assert pooling in ("mean", "max", "instance_max")
        self.pooling = pooling

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        # bag: (N_instance, in_dim)
        if self.pooling == "instance_max":
            h = self.encoder(bag)                    # (N, hidden)
            inst_logits = self.classifier(h)          # (N, n_classes)
            logits, _ = inst_logits.max(dim=0, keepdim=True)   # (1, n_classes)
            return logits

        h = self.encoder(bag)                         # (N, hidden)
        if self.pooling == "mean":
            z = h.mean(dim=0, keepdim=True)
        else:  # "max"
            z, _ = h.max(dim=0, keepdim=True)
        return self.classifier(z)                      # (1, n_classes)


class CNNMILWrapper:
    """shared_functions_V1 이 요구하는 predict_bag() 인터페이스."""
    def __init__(self, model: nn.Module, device: str):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def predict_bag(self, bag_instances) -> dict:
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.model(x)                          # (1, n_classes)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        return {"pred_score": pred_score, "pred_label": pred_label}


# ──────────────────────────────────────────────────────────────
# 2. 학습/평가 루프
# ──────────────────────────────────────────────────────────────

def evaluate(model, bags, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bag in bags:
            x = bag.instances.to(device).float()
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            y_true.append(bag.true_label)
            y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1m


def train_model(model, train_bags, val_bags, device, epochs, lr, weight_decay,
                 class_weights, patience=15):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1, best_state, no_improve = -1.0, None, 0

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
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_bags, device)
        avg_loss = total_loss / len(train_bags)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            log(f"  epoch {epoch:3d}/{epochs} | train_loss {avg_loss:.4f} "
                f"| val_acc {val_acc:.3f} | val_f1_macro {val_f1:.3f} "
                f"| best_f1 {best_val_f1:.3f}")

        if no_improve >= patience:
            log(f"  early stopping @ epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    return model, best_val_f1


# ──────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN-MIL 5-class — holdout 기반")
    parser.add_argument("--pooling", choices=["mean", "max", "instance_max"], default="mean")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="train 내부에서 떼어낼 validation 비율 (early stopping용)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default=None,
                        help="기본값: cnn_mil_v1_{pooling}")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model_name = args.model_name or f"cnn_mil_v1_{args.pooling}"
    model_gen = "gen2_cnn"

    log("=" * 55)
    log("START: CNN-MIL Subtype Classification (5-class)")
    log(f"  Project  : {PROJECT_DIR}")
    log(f"  Device   : {device}")
    log(f"  Pooling  : {args.pooling}")
    log(f"  Classes  : {CLASS_NAMES}")
    log("=" * 55)

    # ── Step 1: 캐시된 임베딩 로드 ──────────────────────────────
    log("[Step 1/7] 캐시된 CNN 임베딩 로드 중...")
    all_dirs = list_patient_dirs(ORGANIZED_DIR)
    all_folders = [d.name for d in all_dirs]
    all_bags = build_bag_objects(all_folders)
    feat_dim = all_bags[0].instances.shape[1]
    log(f"[Step 1/7] 완료 — 전체 환자: {len(all_bags)}명, 임베딩 차원: {feat_dim}")

    # ── Step 2: holdout 분리 ───────────────────────────────────
    log("[Step 2/7] holdout 분리 중...")
    holdout_folders = load_holdout_folders()
    train_val_bags = [b for b in all_bags if b.patient_id not in holdout_folders]
    holdout_bags   = [b for b in all_bags if b.patient_id in holdout_folders]

    missing = holdout_folders - {b.patient_id for b in holdout_bags}
    if missing:
        log(f"  [WARN] holdout_patients.txt에 있지만 organized_data에 없는 폴더: {missing}")

    y_train_val = np.array([b.true_label for b in train_val_bags])
    y_holdout   = np.array([b.true_label for b in holdout_bags])
    log(f"[Step 2/7] 완료 — Train+Val: {len(train_val_bags)}명 / Holdout(test): {len(holdout_bags)}명")
    print_distribution("Train+Val (학습)", y_train_val)
    print_distribution("Test  (holdout)", y_holdout)

    # ── Step 3: train / internal-val 분리 (early stopping용) ──
    log("[Step 3/7] train/val 내부 분리 중...")
    idx = np.arange(len(train_val_bags))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_ratio, stratify=y_train_val, random_state=args.seed,
    )
    train_bags = [train_val_bags[i] for i in train_idx]
    val_bags   = [train_val_bags[i] for i in val_idx]
    log(f"[Step 3/7] 완료 — Train: {len(train_bags)}명 / Val(내부): {len(val_bags)}명")

    # ── Step 4: 모델 학습 ──────────────────────────────────────
    log("[Step 4/7] CNN-MIL 학습 시작...")
    class_weights = compute_class_weights(np.array([b.true_label for b in train_bags]))
    log(f"  class_weights: {class_weights.tolist()}")

    model = CNNMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim,
                    n_classes=N_CLASSES, pooling=args.pooling, dropout=args.dropout)
    t0 = time.time()
    model, best_val_f1 = train_model(
        model, train_bags, val_bags, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, patience=args.patience,
    )
    log(f"[Step 4/7] 학습 완료 ({time.time()-t0:.1f}s, best internal val F1_macro={best_val_f1:.3f})")

    # ── Step 5: 모델 저장 ──────────────────────────────────────
    log("[Step 5/7] 모델 저장 중...")
    model_dir = MODEL_ROOT / model_gen
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": feat_dim,
        "hidden_dim": args.hidden_dim,
        "n_classes": N_CLASSES,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "classes": CLASS_NAMES,
    }, model_path)
    log(f"[Step 5/7] 모델 저장 완료 → {model_path}")

    # ── Step 6: shared_functions 평가 ─────────────────────────
    log("[Step 6/7] holdout 평가 중 (shared_functions)...")
    wrapper = CNNMILWrapper(model, device)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = wrapper,
        holdout_bags = holdout_bags,
        model_gen    = model_gen,
        model_name   = model_name,
        output_dir   = str(OUTPUT_DIR),
    )
    log("[Step 6/7] shared_functions 평가 완료")

    # ── Step 7: 5-class 상세 결과 출력 및 저장 ────────────────
    log("[Step 7/7] 5-class 상세 결과 계산 중...")
    y_true = y_holdout
    y_pred = np.array([
        bag_df.loc[bag_df["patient_id"] == b.patient_id, "pred_label"].values[0]
        for b in holdout_bags
    ])

    f1_per_cls  = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                           average=None, zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true, y_pred)
    cm          = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"Holdout 평가 결과 (5-class) — CNN-MIL ({args.pooling})", flush=True)
    print("=" * 60, flush=True)
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
        print(f"  {CLASS_NAMES[i][:10]:<12}" + "  ".join(f"{v:>6}" for v in row), flush=True)

    report = classification_report(
        y_true, y_pred, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, zero_division=0,
    )
    print(f"\n  Classification report:\n{report}", flush=True)

    perf_dir = OUTPUT_DIR / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    detail = {
        "model_gen":    model_gen,
        "model_name":   model_name,
        "pooling":      args.pooling,
        "n_train":      int(len(train_bags)),
        "n_val_internal": int(len(val_bags)),
        "n_test":       int(len(holdout_bags)),
        "best_internal_val_f1_macro": float(best_val_f1),
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]) for cls in CLASS_NAMES},
    }
    detail_path = perf_dir / f"performance_metrics_5class.{model_gen}_{model_name}.json"
    detail_path.write_text(json.dumps(detail, indent=2))

    log("[Step 7/7] 완료")
    log("=" * 55)
    log("DONE: 모든 단계 완료")
    log(f"  model       : {model_path}")
    log(f"  performance : {detail_path}")
    log("=" * 55)
    print("\nshared_functions metrics_df:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
