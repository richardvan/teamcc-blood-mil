#!/usr/bin/env python3
"""
07_2_attention_mil_subtype_cv_holdout.py — Attention-MIL, metadata.csv 기반 5-fold CV + holdout

07_1 과의 차이:
  - holdout_data/holdout_patients.txt 대신 metadata.csv 단일 소스 사용
    (is_holdout=True 인 환자 = 최종 holdout, 나머지는 fold_1~5_status 로 5-fold 구성)
  - "랜덤 15% val" 한 번 대신, metadata에 미리 stratified 되어 있는 5-fold로
    5번 학습/평가를 반복 → 평균±표준편차로 성능의 안정성을 확인
    (표본이 2~3개뿐인 소수 클래스의 노이즈를 좀 더 잘 드러냄)
  - 최종 모델은 non-holdout 161명 전체로 학습해서 진짜 holdout(28명)에 대해서만
    최종 리포트 (이 부분은 07_1과 동일한 방식)

전제 조건:
  - metadata.csv 가 PROJECT_DIR (blood_mil_project/) 바로 아래 있어야 함
  - python 00_extract_cnn_features.py 로 cache/cnn_features/*.pt 준비되어 있어야 함

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_2_attention_mil_subtype_cv_holdout.py --epochs 60 --top_k 10
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
    log, print_distribution, compute_class_weights, get_instance_filenames,
    build_bag_objects, parse_label_from_folder,
    load_metadata, get_holdout_folders_from_metadata, get_fold_split,
    FEAT_CACHE_DIR,
)

from shared_functions_V2 import predict_labels_and_report_performance


# ──────────────────────────────────────────────────────────────
# 1. 모델 정의 (Gated Attention MIL, Ilse et al. 2018) — 07_1과 동일
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
        h = self.encoder(bag)                        # (N, hidden)

        a_v = self.attn_V(h)                          # (N, attn_dim)
        if self.gated:
            a_u = self.attn_U(h)
            scores = self.attn_w(a_v * a_u)           # (N, 1)
        else:
            scores = self.attn_w(a_v)

        attn_weights = torch.softmax(scores, dim=0)    # (N, 1)
        z = (attn_weights * h).sum(dim=0, keepdim=True)  # (1, hidden)

        logits = self.classifier(z)                     # (1, n_classes)
        return logits, attn_weights.squeeze(1)           # (1,n_classes), (N,)


class AttentionMILWrapper:
    """shared_functions_V2 이 요구하는 predict_bag() 인터페이스."""
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
# 2. 학습/평가 루프 (07_1과 동일)
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
    return acc, f1m, np.array(y_true), np.array(y_pred)


def train_model(model, train_bags, val_bags, device, epochs, lr, weight_decay,
                 class_weights, patience=15, verbose=True):
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

        val_acc, val_f1, _, _ = evaluate(model, val_bags, device)
        avg_loss = total_loss / len(train_bags)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            log(f"    epoch {epoch:3d}/{epochs} | train_loss {avg_loss:.4f} "
                f"| val_acc {val_acc:.3f} | val_f1_macro {val_f1:.3f} "
                f"| best_f1 {best_val_f1:.3f}")

        if no_improve >= patience:
            if verbose:
                log(f"    early stopping @ epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    return model, best_val_f1


def export_top_attended_cells(wrapper, holdout_bags, top_k, output_dir):
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
    out_path = pred_dir / "attention_top_cells.gen3_attention_attention_mil_cv_v1.csv"
    df.to_csv(out_path, index=False)
    log(f"  attention top-{top_k} 세포 저장 → {out_path}")
    return out_path


def make_bag_lookup(all_folders):
    """전체 폴더에 대한 folder -> BagObject 딕셔너리 (fold마다 재사용, 캐시 재로딩 방지)."""
    bags = build_bag_objects(all_folders)
    return {b.patient_id: b for b in bags}


# ──────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention-MIL 5-class (metadata 5-fold CV + holdout)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--gated", action="store_true", default=True)
    parser.add_argument("--no_gated", dest="gated", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--internal_val_ratio", type=float, default=0.15,
                        help="각 fold의 train 안에서 early stopping용으로 떼는 비율")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv",
                        help="PROJECT_DIR 바로 아래 위치한 metadata csv 파일명")
    parser.add_argument("--model_name", type=str, default="attention_mil_cv_v1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model_gen = "gen3_attention"

    log("=" * 55)
    log("START: Attention-MIL (metadata 5-fold CV + holdout)")
    log(f"  Project  : {PROJECT_DIR}")
    log(f"  Device   : {device}")
    log(f"  Classes  : {CLASS_NAMES}")
    log("=" * 55)

    # ── Step 1: metadata 로드 ───────────────────────────────────
    log(f"[Step 1/9] {args.metadata_file} 로드 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = get_holdout_folders_from_metadata(meta)
    log(f"[Step 1/9] 완료 — 전체 {len(meta)}명, holdout {len(holdout_folders)}명")

    # ── Step 2: 캐시된 임베딩 전체 로드 (fold마다 재사용) ────────
    log("[Step 2/9] 캐시된 CNN 임베딩 로드 중...")
    bag_lookup = make_bag_lookup(meta["folder"].tolist())
    feat_dim = next(iter(bag_lookup.values())).instances.shape[1]
    log(f"[Step 2/9] 완료 — 임베딩 차원: {feat_dim}")

    # ── Step 3: 5-fold CV ───────────────────────────────────────
    log(f"[Step 3/9] {args.n_folds}-fold CV 시작...")
    cv_results = []
    for fold in range(1, args.n_folds + 1):
        log(f"  --- Fold {fold}/{args.n_folds} ---")
        train_folders, test_folders = get_fold_split(meta, fold)

        # fold의 train 안에서 다시 internal val 떼기 (early stopping용)
        y_train_all = np.array([parse_label_from_folder(f) for f in train_folders])
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_folders)), test_size=args.internal_val_ratio,
            stratify=y_train_all, random_state=args.seed,
        )
        fold_train_bags = [bag_lookup[train_folders[i]] for i in tr_idx]
        fold_val_bags   = [bag_lookup[train_folders[i]] for i in val_idx]
        fold_test_bags  = [bag_lookup[f] for f in test_folders]

        class_weights = compute_class_weights(np.array([b.true_label for b in fold_train_bags]))
        model = AttentionMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
                              n_classes=N_CLASSES, dropout=args.dropout, gated=args.gated)
        model, _ = train_model(
            model, fold_train_bags, fold_val_bags, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            class_weights=class_weights, patience=args.patience, verbose=False,
        )

        test_acc, test_f1, y_true, y_pred = evaluate(model, fold_test_bags, device)
        f1_per_cls = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                               average=None, zero_division=0)
        log(f"  Fold {fold}: n_test={len(fold_test_bags)} | acc={test_acc:.3f} | f1_macro={test_f1:.3f}")
        cv_results.append({
            "fold": fold, "n_test": len(fold_test_bags),
            "accuracy": test_acc, "f1_macro": test_f1,
            "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]) for cls in CLASS_NAMES},
        })

    cv_acc  = np.array([r["accuracy"] for r in cv_results])
    cv_f1m  = np.array([r["f1_macro"] for r in cv_results])
    log(f"[Step 3/9] CV 완료 — accuracy {cv_acc.mean():.3f}±{cv_acc.std():.3f} "
        f"| f1_macro {cv_f1m.mean():.3f}±{cv_f1m.std():.3f}")

    # ── Step 4: 최종 모델 학습 (non-holdout 전체) ───────────────
    log("[Step 4/9] 최종 모델 학습 시작 (non-holdout 전체 사용)...")
    train_pool_folders = meta.loc[~meta["folder"].isin(holdout_folders), "folder"].tolist()
    y_pool = np.array([parse_label_from_folder(f) for f in train_pool_folders])
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_pool_folders)), test_size=args.internal_val_ratio,
        stratify=y_pool, random_state=args.seed,
    )
    final_train_bags = [bag_lookup[train_pool_folders[i]] for i in tr_idx]
    final_val_bags   = [bag_lookup[train_pool_folders[i]] for i in val_idx]
    holdout_bags     = [bag_lookup[f] for f in holdout_folders]

    print_distribution("최종 Train", np.array([b.true_label for b in final_train_bags]))
    print_distribution("최종 내부 Val", np.array([b.true_label for b in final_val_bags]))
    print_distribution("Holdout (test)", np.array([b.true_label for b in holdout_bags]))

    class_weights = compute_class_weights(np.array([b.true_label for b in final_train_bags]))
    model = AttentionMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
                          n_classes=N_CLASSES, dropout=args.dropout, gated=args.gated)
    t0 = time.time()
    model, best_val_f1 = train_model(
        model, final_train_bags, final_val_bags, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, patience=args.patience, verbose=True,
    )
    log(f"[Step 4/9] 학습 완료 ({time.time()-t0:.1f}s, best internal val F1_macro={best_val_f1:.3f})")

    # ── Step 5: 모델 저장 ──────────────────────────────────────
    log("[Step 5/9] 모델 저장 중...")
    model_dir = MODEL_ROOT / model_gen
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": feat_dim, "hidden_dim": args.hidden_dim, "attn_dim": args.attn_dim,
        "gated": args.gated, "n_classes": N_CLASSES,
        "dropout": args.dropout, "classes": CLASS_NAMES,
    }, model_path)
    log(f"[Step 5/9] 모델 저장 완료 → {model_path}")

    # ── Step 6: shared_functions 평가 (holdout) ────────────────
    log("[Step 6/9] holdout 평가 중 (shared_functions_V2)...")
    wrapper = AttentionMILWrapper(model, device)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = wrapper,
        holdout_bags = holdout_bags,
        model_gen    = model_gen,
        model_name   = args.model_name,
        output_dir   = str(OUTPUT_DIR),
    )
    log("[Step 6/9] shared_functions 평가 완료")

    # ── Step 7: attention 해석용 결과 저장 ─────────────────────
    log("[Step 7/9] top-attention 세포 저장 중...")
    export_top_attended_cells(wrapper, holdout_bags, args.top_k, OUTPUT_DIR)
    log("[Step 7/9] 완료")

    # ── Step 8: 5-class 상세 holdout 결과 ──────────────────────
    log("[Step 8/9] 5-class 상세 결과 계산 중...")
    y_true = np.array([b.true_label for b in holdout_bags])
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
    print("5-fold CV 요약", flush=True)
    print("=" * 60, flush=True)
    for r in cv_results:
        print(f"  Fold {r['fold']}: n_test={r['n_test']:2d} | "
              f"acc={r['accuracy']:.3f} | f1_macro={r['f1_macro']:.3f}", flush=True)
    print(f"  평균        : acc={cv_acc.mean():.3f}±{cv_acc.std():.3f} | "
          f"f1_macro={cv_f1m.mean():.3f}±{cv_f1m.std():.3f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("최종 Holdout 평가 결과 (5-class) — Attention-MIL", flush=True)
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

    # ── Step 9: 결과 저장 (CV + holdout 통합 json) ─────────────
    perf_dir = OUTPUT_DIR / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    detail = {
        "model_gen":  model_gen,
        "model_name": args.model_name,
        "cv": {
            "n_folds": args.n_folds,
            "per_fold": cv_results,
            "mean_accuracy": float(cv_acc.mean()), "std_accuracy": float(cv_acc.std()),
            "mean_f1_macro": float(cv_f1m.mean()), "std_f1_macro": float(cv_f1m.std()),
        },
        "holdout": {
            "n_train":        int(len(final_train_bags)),
            "n_val_internal":  int(len(final_val_bags)),
            "n_test":          int(len(holdout_bags)),
            "best_internal_val_f1_macro": float(best_val_f1),
            "accuracy":     float(acc),
            "f1_macro":     float(f1_macro),
            "f1_weighted":  float(f1_weighted),
            "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]) for cls in CLASS_NAMES},
        },
    }
    detail_path = perf_dir / f"performance_metrics_5class.{model_gen}_{args.model_name}.json"
    detail_path.write_text(json.dumps(detail, indent=2))

    log("[Step 9/9] 완료")
    log("=" * 55)
    log("DONE: 모든 단계 완료")
    log(f"  model       : {model_path}")
    log(f"  performance : {detail_path}")
    log("=" * 55)
    print("\nshared_functions metrics_df:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
