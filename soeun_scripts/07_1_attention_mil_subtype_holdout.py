#!/usr/bin/env python3
"""
Attention-MIL: AML Subtype Classification (5-class) — holdout 기반, GPU 학습

CNN-MIL과의 차이:
  - mean/max pooling(모든 인스턴스를 동등하게 취급) 대신
    Ilse et al. (2018) "Attention-based Deep MIL"의 gated-attention pooling 사용.
  - 인스턴스(세포)마다 attention weight를 학습 → 진단에 중요한 세포에
    자동으로 더 큰 가중치를 준 뒤 가중합으로 bag 벡터를 만듦.
  - 부가 효과: 어떤 세포 이미지가 예측에 가장 크게 기여했는지 확인 가능
    (해석/시각화, 병리 전문가 검증에 유용) → holdout 평가 후
    환자별 top-k attention 세포를 predictions/attention_top_cells.*.csv 로 저장.

전제 조건:
  python 00_extract_cnn_features.py 를 먼저 실행해 cache/cnn_features/*.pt 준비

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_1_attention_mil_subtype_holdout.py --epochs 60 --top_k 10
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
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
    BagObject, build_bag_objects, compute_class_weights,
    get_instance_filenames, ORGANIZED_DIR,
)

from shared_functions_V1 import predict_labels_and_report_performance


# ──────────────────────────────────────────────────────────────
# 1. 모델 정의 (Gated Attention MIL, Ilse et al. 2018)
# ──────────────────────────────────────────────────────────────

class AttentionMIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, attn_dim: int, n_classes: int,
                 dropout: float = 0.3, gated: bool = True):
        super().__init__()
        self.gated = gated

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        if gated:
            self.attn_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, bag: torch.Tensor):
        """
        bag: (N_instance, in_dim)
        반환: logits (1, n_classes), attn_weights (N_instance,) — softmax 정규화됨
        """
        h = self.encoder(bag)                       # (N, hidden)

        a_v = self.attn_V(h)                        # (N, attn_dim)
        if self.gated:
            a_u = self.attn_U(h)                     # (N, attn_dim)
            scores = self.attn_w(a_v * a_u)          # (N, 1)
        else:
            scores = self.attn_w(a_v)                # (N, 1)

        attn_weights = torch.softmax(scores, dim=0)   # (N, 1) — 인스턴스 축으로 정규화
        z = (attn_weights * h).sum(dim=0, keepdim=True)   # (1, hidden) — 가중합

        logits = self.classifier(z)                   # (1, n_classes)
        return logits, attn_weights.squeeze(1)         # (1,n_classes), (N,)


class AttentionMILWrapper:
    """shared_functions_V1 이 요구하는 predict_bag() 인터페이스."""
    def __init__(self, model: nn.Module, device: str):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def predict_bag(self, bag_instances) -> dict:
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        return {"pred_score": pred_score, "pred_label": pred_label}

    @torch.no_grad()
    def predict_bag_with_attention(self, bag_instances):
        """attention 가중치까지 함께 반환 (해석용, shared_functions에서는 안 씀)."""
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, attn = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        return {
            "pred_score": pred_score,
            "pred_label": pred_label,
            "attn_weights": attn.cpu().numpy(),
        }


# ──────────────────────────────────────────────────────────────
# 2. 학습/평가 루프
# ──────────────────────────────────────────────────────────────

def evaluate(model, bags, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bag in bags:
            x = bag.instances.to(device).float()
            logits, _ = model(x)
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
            logits, _ = model(x)
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


def export_top_attended_cells(wrapper, holdout_bags, top_k, output_dir):
    """환자별로 attention이 가장 높았던 top-k 세포 이미지 파일명을 저장 (해석/시각화용)."""
    rows = []
    for bag in holdout_bags:
        result = wrapper.predict_bag_with_attention(bag.instances)
        attn = result["attn_weights"]
        filenames = get_instance_filenames(bag.patient_id)

        if len(filenames) != len(attn):
            log(f"  [WARN] {bag.patient_id}: 파일명 수({len(filenames)}) != "
                f"인스턴스 수({len(attn)}) — 캐시 재추출 필요 가능성")
            continue

        top_idx = np.argsort(-attn)[:top_k]
        for rank, i in enumerate(top_idx, start=1):
            rows.append({
                "patient_id": bag.patient_id,
                "true_label": LABEL_TO_SUBTYPE[bag.true_label],
                "pred_label": LABEL_TO_SUBTYPE[result["pred_label"]],
                "rank": rank,
                "filename": filenames[i],
                "attn_weight": float(attn[i]),
            })

    df = pd.DataFrame(rows)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "attention_top_cells.gen3_attention_attention_mil_v1.csv"
    df.to_csv(out_path, index=False)
    log(f"  attention top-{top_k} 세포 저장 → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention-MIL 5-class — holdout 기반")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--gated", action="store_true", default=True)
    parser.add_argument("--no_gated", dest="gated", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--top_k", type=int, default=10,
                        help="환자별로 저장할 top attention 세포 개수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="attention_mil_v1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model_gen = "gen3_attention"

    log("=" * 55)
    log("START: Attention-MIL Subtype Classification (5-class)")
    log(f"  Project  : {PROJECT_DIR}")
    log(f"  Device   : {device}")
    log(f"  Gated    : {args.gated}")
    log(f"  Classes  : {CLASS_NAMES}")
    log("=" * 55)

    # ── Step 1: 캐시된 임베딩 로드 ──────────────────────────────
    log("[Step 1/8] 캐시된 CNN 임베딩 로드 중...")
    all_dirs = list_patient_dirs(ORGANIZED_DIR)
    all_folders = [d.name for d in all_dirs]
    all_bags = build_bag_objects(all_folders)
    feat_dim = all_bags[0].instances.shape[1]
    log(f"[Step 1/8] 완료 — 전체 환자: {len(all_bags)}명, 임베딩 차원: {feat_dim}")

    # ── Step 2: holdout 분리 ───────────────────────────────────
    log("[Step 2/8] holdout 분리 중...")
    holdout_folders = load_holdout_folders()
    train_val_bags = [b for b in all_bags if b.patient_id not in holdout_folders]
    holdout_bags   = [b for b in all_bags if b.patient_id in holdout_folders]

    missing = holdout_folders - {b.patient_id for b in holdout_bags}
    if missing:
        log(f"  [WARN] holdout_patients.txt에 있지만 organized_data에 없는 폴더: {missing}")

    y_train_val = np.array([b.true_label for b in train_val_bags])
    y_holdout   = np.array([b.true_label for b in holdout_bags])
    log(f"[Step 2/8] 완료 — Train+Val: {len(train_val_bags)}명 / Holdout(test): {len(holdout_bags)}명")
    print_distribution("Train+Val (학습)", y_train_val)
    print_distribution("Test  (holdout)", y_holdout)

    # ── Step 3: train / internal-val 분리 ──────────────────────
    log("[Step 3/8] train/val 내부 분리 중...")
    idx = np.arange(len(train_val_bags))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_ratio, stratify=y_train_val, random_state=args.seed,
    )
    train_bags = [train_val_bags[i] for i in train_idx]
    val_bags   = [train_val_bags[i] for i in val_idx]
    log(f"[Step 3/8] 완료 — Train: {len(train_bags)}명 / Val(내부): {len(val_bags)}명")

    # ── Step 4: 모델 학습 ──────────────────────────────────────
    log("[Step 4/8] Attention-MIL 학습 시작...")
    class_weights = compute_class_weights(np.array([b.true_label for b in train_bags]))
    log(f"  class_weights: {class_weights.tolist()}")

    model = AttentionMIL(
        in_dim=feat_dim, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
        n_classes=N_CLASSES, dropout=args.dropout, gated=args.gated,
    )
    t0 = time.time()
    model, best_val_f1 = train_model(
        model, train_bags, val_bags, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, patience=args.patience,
    )
    log(f"[Step 4/8] 학습 완료 ({time.time()-t0:.1f}s, best internal val F1_macro={best_val_f1:.3f})")

    # ── Step 5: 모델 저장 ──────────────────────────────────────
    log("[Step 5/8] 모델 저장 중...")
    model_dir = MODEL_ROOT / model_gen
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": feat_dim,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "gated": args.gated,
        "n_classes": N_CLASSES,
        "dropout": args.dropout,
        "classes": CLASS_NAMES,
    }, model_path)
    log(f"[Step 5/8] 모델 저장 완료 → {model_path}")

    # ── Step 6: shared_functions 평가 ─────────────────────────
    log("[Step 6/8] holdout 평가 중 (shared_functions)...")
    wrapper = AttentionMILWrapper(model, device)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = wrapper,
        holdout_bags = holdout_bags,
        model_gen    = model_gen,
        model_name   = args.model_name,
        output_dir   = str(OUTPUT_DIR),
    )
    log("[Step 6/8] shared_functions 평가 완료")

    # ── Step 7: attention 해석용 결과 저장 ─────────────────────
    log("[Step 7/8] top-attention 세포 저장 중...")
    export_top_attended_cells(wrapper, holdout_bags, args.top_k, OUTPUT_DIR)
    log("[Step 7/8] 완료")

    # ── Step 8: 5-class 상세 결과 출력 및 저장 ────────────────
    log("[Step 8/8] 5-class 상세 결과 계산 중...")
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
    print("Holdout 평가 결과 (5-class) — Attention-MIL", flush=True)
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
        "model_name":   args.model_name,
        "gated":        args.gated,
        "n_train":      int(len(train_bags)),
        "n_val_internal": int(len(val_bags)),
        "n_test":       int(len(holdout_bags)),
        "best_internal_val_f1_macro": float(best_val_f1),
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]) for cls in CLASS_NAMES},
    }
    detail_path = perf_dir / f"performance_metrics_5class.{model_gen}_{args.model_name}.json"
    detail_path.write_text(json.dumps(detail, indent=2))

    log("[Step 8/8] 완료")
    log("=" * 55)
    log("DONE: 모든 단계 완료")
    log(f"  model       : {model_path}")
    log(f"  performance : {detail_path}")
    log("=" * 55)
    print("\nshared_functions metrics_df:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
