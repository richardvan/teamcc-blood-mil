#!/usr/bin/env python3
"""
07_2_attention_mil_train.py — Attention-MIL 학습 (holdout 제외, CV + 최종 모델)

팀원의 j_SVM_multiclass.py 와 동일한 산출물 형태를 목표로 합니다:
  - classification_report.txt  (fold별 리포트 + 전체 aggregate 리포트)
  - confusion_matrix.txt        (aggregate confusion matrix)
  - pca_plot.png                (환자별 bag 벡터를 PCA로 2차원 시각화)
  - robustness 체크 (여러 seed로 반복 학습: 실제 라벨 vs 셔플 라벨 정확도 비교
    → 셔플 라벨인데도 정확도가 높게 나오면 데이터 누수 의심)
  - svm_best_model.joblib 대신 attention_mil_v1.pt 로 모델 저장

SVM과의 구조적 차이:
  - SVM은 "환자당 이미지들을 미리 평균 낸 벡터 1개"로 압축한 뒤 SVC를 학습시키지만,
    Attention-MIL은 pooling(가중합) 자체를 모델이 학습합니다.
    PCA 시각화에서 "환자당 벡터"에 대응하는 게 SVM은 mean-pooled feature,
    Attention-MIL은 학습된 attention으로 가중합한 bag 벡터(z)입니다.
  - CV/holdout 분리는 metadata_for_multiclass.csv 의 is_holdout / fold_k_status
    컬럼을 그대로 사용합니다 (팀원 코드의 holdout_data_for_multiclass/,
    cv_splits_for_multiclass/ 폴더와 동일한 정보를 담고 있는 컬럼 버전).

전제 조건:
  - metadata_for_multiclass.csv 가 PROJECT_DIR 바로 아래 있어야 함
  - python 00_extract_cnn_features.py 로 cache/cnn_features/*.pt 준비되어 있어야 함

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_2_attention_mil_train.py --epochs 60
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 서버(디스플레이 없음)에서 안전하게 그림 저장
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, classification_report,
)

from mil_common import (
    PROJECT_DIR, MODEL_ROOT, CLASS_NAMES, N_CLASSES,
    log, print_distribution, compute_class_weights,
    build_bag_objects, parse_label_from_folder,
    load_metadata, get_holdout_folders_from_metadata, get_fold_split,
)
from attention_mil_common import (
    AttentionMIL, AttentionMILWrapper, evaluate, train_model, relabel_bags,
    compute_auc_macro, compute_auc_per_class, compute_roc_curve_points,
)


def make_bag_lookup(all_folders):
    bags = build_bag_objects(all_folders)
    return {b.patient_id: b for b in bags}


def build_model(args, feat_dim):
    return AttentionMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
                         n_classes=N_CLASSES, dropout=args.dropout, gated=args.gated)


def quick_eval(bags, seed, args, device, shuffle_labels=False, epochs=30):
    """
    팀원의 quick_eval() 과 동일한 취지: 랜덤 80/20 split 한 번으로 빠르게 학습/평가.
    real vs shuffled 라벨 비교용이라 early stopping 없이 고정 epoch만 돕니다
    (test set을 val로도 재사용하면 조기종료 자체가 살짝 낙관적으로 새서, 이 체크에서는 뺐습니다).
    """
    labels = np.array([b.true_label for b in bags])
    if shuffle_labels:
        rng = np.random.RandomState(seed)
        labels = rng.permutation(labels)

    idx = np.arange(len(bags))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)

    relabeled = relabel_bags(bags, labels)
    train_bags = [relabeled[i] for i in train_idx]
    test_bags  = [relabeled[i] for i in test_idx]

    class_weights = compute_class_weights(np.array([b.true_label for b in train_bags]))
    model = build_model(args, train_bags[0].instances.shape[1])
    model, _ = train_model(
        model, train_bags, test_bags, device,
        epochs=epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, verbose=False, use_early_stopping=False,
    )
    _, _, y_true, y_pred, _ = evaluate(model, test_bags, device)
    return balanced_accuracy_score(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Attention-MIL 학습 (CV + 최종 모델 + robustness + PCA)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--gated", action="store_true", default=True)
    parser.add_argument("--no_gated", dest="gated", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--internal_val_ratio", type=float, default=0.15)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_robustness_seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--model_name", type=str, default="attention_mil_v1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model_gen = "gen3_attention"

    save_dir = MODEL_ROOT / model_gen / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: Attention-MIL 학습 (CV + 최종 모델)")
    log(f"  Device : {device}")
    log(f"  SaveDir: {save_dir}")
    log("=" * 55)

    # ── metadata / 캐시 로드 ────────────────────────────────────
    log(f"[1] {args.metadata_file} 로드 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = get_holdout_folders_from_metadata(meta)
    bag_lookup = make_bag_lookup(meta["folder"].tolist())
    feat_dim = next(iter(bag_lookup.values())).instances.shape[1]
    log(f"[1] 완료 — 전체 {len(meta)}명, holdout {len(holdout_folders)}명 (학습에서 완전히 제외)")

    train_pool_folders = meta.loc[~meta["folder"].isin(holdout_folders), "folder"].tolist()

    # ── 5-fold CV (aggregate report 생성용) ─────────────────────
    log(f"[2] {args.n_folds}-fold CV 시작...")
    fold_reports = []
    agg_y_true, agg_y_pred = [], []
    agg_y_proba = []
    fold_aucs = []

    for fold in range(1, args.n_folds + 1):
        train_folders, test_folders = get_fold_split(meta, fold)
        y_train_all = np.array([parse_label_from_folder(f) for f in train_folders])
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_folders)), test_size=args.internal_val_ratio,
            stratify=y_train_all, random_state=args.seed,
        )
        fold_train_bags = [bag_lookup[train_folders[i]] for i in tr_idx]
        fold_val_bags   = [bag_lookup[train_folders[i]] for i in val_idx]
        fold_test_bags  = [bag_lookup[f] for f in test_folders]

        class_weights = compute_class_weights(np.array([b.true_label for b in fold_train_bags]))
        model = build_model(args, feat_dim)
        model, _ = train_model(
            model, fold_train_bags, fold_val_bags, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            class_weights=class_weights, patience=args.patience, verbose=False,
        )
        _, _, y_true, y_pred, y_proba = evaluate(model, fold_test_bags, device)
        agg_y_true.extend(y_true.tolist())
        agg_y_pred.extend(y_pred.tolist())
        agg_y_proba.extend(y_proba.tolist())

        fold_auc = compute_auc_macro(y_true, y_proba, N_CLASSES)
        fold_aucs.append(fold_auc)

        fold_report = classification_report(
            y_true, y_pred, labels=list(range(N_CLASSES)),
            target_names=CLASS_NAMES, zero_division=0,
        )
        fold_reports.append((fold, fold_report))
        bacc = balanced_accuracy_score(y_true, y_pred)
        log(f"  Fold {fold}: n_test={len(fold_test_bags)} | balanced_acc={bacc:.3f} | auc_macro={fold_auc:.3f}")

    agg_y_true  = np.array(agg_y_true)
    agg_y_pred  = np.array(agg_y_pred)
    agg_y_proba = np.array(agg_y_proba)
    agg_bacc = balanced_accuracy_score(agg_y_true, agg_y_pred)
    agg_f1   = f1_score(agg_y_true, agg_y_pred, average="macro", zero_division=0)
    agg_auc  = compute_auc_macro(agg_y_true, agg_y_proba, N_CLASSES)
    agg_auc_per_class = compute_auc_per_class(agg_y_true, agg_y_proba, N_CLASSES)
    log(f"[2] CV 완료 — aggregate balanced_acc={agg_bacc:.3f} | f1_macro={agg_f1:.3f} | auc_macro={agg_auc:.3f}")

    # classification_report.txt (팀원 코드와 동일한 구성: 각 fold + aggregate)
    report_path = save_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"=== Attention-MIL hyperparameters ===\n")
        f.write(json.dumps(vars(args), indent=2) + "\n\n")
        for fold, rep in fold_reports:
            f.write(f"=== Fold {fold} ===\n{rep}\n")
        f.write("=== Aggregate (all folds combined) ===\n")
        f.write(classification_report(
            agg_y_true, agg_y_pred, labels=list(range(N_CLASSES)),
            target_names=CLASS_NAMES, zero_division=0,
        ))
    log(f"  classification_report 저장 → {report_path}")

    cm_path = save_dir / "confusion_matrix.txt"
    cm = confusion_matrix(agg_y_true, agg_y_pred, labels=list(range(N_CLASSES)))
    np.savetxt(cm_path, cm, fmt="%d")
    log(f"  confusion_matrix 저장 → {cm_path}")

    auc_path = save_dir / "auc_metrics.json"
    auc_path.write_text(json.dumps({
        "per_fold_auc_macro": fold_aucs,
        "aggregate_auc_macro": agg_auc,
        "aggregate_auc_per_class": {CLASS_NAMES[c]: v for c, v in agg_auc_per_class.items()},
    }, indent=2))
    log(f"  AUC 결과 저장 → {auc_path}")

    roc_df = compute_roc_curve_points(agg_y_true, agg_y_proba, N_CLASSES, CLASS_NAMES)
    roc_path = save_dir / "roc_curve_points.csv"
    roc_df.to_csv(roc_path, index=False)
    log(f"  ROC curve 좌표 저장 → {roc_path}")

    plt.figure()
    for cls_name, g in roc_df.groupby("class"):
        auc_val = agg_auc_per_class[CLASS_NAMES.index(cls_name)]
        plt.plot(g["fpr"], g["tpr"], label=f"{cls_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves (5-fold CV aggregate, macro AUC={agg_auc:.3f})")
    plt.legend(fontsize=8)
    roc_plot_path = save_dir / "roc_curve_plot.png"
    plt.savefig(roc_plot_path, bbox_inches="tight")
    plt.close()
    log(f"  ROC curve plot 저장 → {roc_plot_path}")

    # ── 최종 모델 학습 (non-holdout 전체) ───────────────────────
    log("[3] 최종 모델 학습 시작 (non-holdout 전체 사용)...")
    y_pool = np.array([parse_label_from_folder(f) for f in train_pool_folders])
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_pool_folders)), test_size=args.internal_val_ratio,
        stratify=y_pool, random_state=args.seed,
    )
    final_train_bags = [bag_lookup[train_pool_folders[i]] for i in tr_idx]
    final_val_bags   = [bag_lookup[train_pool_folders[i]] for i in val_idx]

    print_distribution("최종 Train", np.array([b.true_label for b in final_train_bags]))
    print_distribution("최종 내부 Val", np.array([b.true_label for b in final_val_bags]))

    class_weights = compute_class_weights(np.array([b.true_label for b in final_train_bags]))
    model = build_model(args, feat_dim)
    model, best_val_f1 = train_model(
        model, final_train_bags, final_val_bags, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, patience=args.patience, verbose=True,
    )
    log(f"[3] 학습 완료 (best internal val F1_macro={best_val_f1:.3f})")

    model_dir = MODEL_ROOT / model_gen
    model_path = model_dir / f"{args.model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": feat_dim, "hidden_dim": args.hidden_dim, "attn_dim": args.attn_dim,
        "gated": args.gated, "n_classes": N_CLASSES,
        "dropout": args.dropout, "classes": CLASS_NAMES,
    }, model_path)
    log(f"  모델 저장 완료 → {model_path}")

    # ── robustness 체크 (real vs shuffled labels) ──────────────
    log(f"[4] robustness 체크 시작 ({args.n_robustness_seeds} seeds, real vs shuffled labels)...")
    train_pool_bags = [bag_lookup[f] for f in train_pool_folders]

    real_scores = [quick_eval(train_pool_bags, s, args, device, shuffle_labels=False)
                   for s in range(args.n_robustness_seeds)]
    shuf_scores = [quick_eval(train_pool_bags, s, args, device, shuffle_labels=True)
                   for s in range(args.n_robustness_seeds)]

    log(f"  real labels : {np.round(real_scores, 3).tolist()} mean {np.mean(real_scores):.3f}")
    log(f"  shuffled    : {np.round(shuf_scores, 3).tolist()} mean {np.mean(shuf_scores):.3f}")
    if np.mean(shuf_scores) > 0.35:   # 5-class 우연 수준(~0.2)보다 뚜렷이 높으면 경고
        log("  [WARN] 셔플 라벨인데도 balanced accuracy가 높습니다 — 데이터 누수 의심, 확인 필요")

    robustness_path = save_dir / "robustness_check.json"
    robustness_path.write_text(json.dumps({
        "real_scores": real_scores, "real_mean": float(np.mean(real_scores)),
        "shuffled_scores": shuf_scores, "shuffled_mean": float(np.mean(shuf_scores)),
    }, indent=2))
    log(f"  robustness 결과 저장 → {robustness_path}")

    # ── PCA 시각화 (환자별 attention-pooled bag 벡터) ──────────
    log("[5] PCA 시각화 준비 중...")
    wrapper = AttentionMILWrapper(model, device)
    groups, bag_vectors, labels = [], [], []
    for folder in train_pool_folders:
        bag = bag_lookup[folder]
        z = wrapper.get_bag_vector(bag.instances)
        groups.append(folder)
        bag_vectors.append(z)
        labels.append(bag.true_label)

    X_bagvec = np.array(bag_vectors)
    groups   = np.array(groups)
    labels   = np.array(labels)

    np.save(save_dir / "X_bagvec.npy", X_bagvec)
    np.save(save_dir / "groups_bagvec.npy", groups)
    np.save(save_dir / "labels_bagvec.npy", labels)

    scaler = StandardScaler().fit(X_bagvec)
    pca = PCA(n_components=2).fit(scaler.transform(X_bagvec))
    Z = pca.transform(scaler.transform(X_bagvec))

    import pandas as pd
    pca_df = pd.DataFrame({
        "patient_id": groups,
        "true_subtype": [CLASS_NAMES[l] for l in labels],
        "pc1": Z[:, 0],
        "pc2": Z[:, 1],
    })
    pca_csv_path = save_dir / "pca_coords.csv"
    pca_df.to_csv(pca_csv_path, index=False)
    log(f"[5] PCA 좌표 저장 → {pca_csv_path}")

    colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
    plt.figure()
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        plt.scatter(Z[mask, 0], Z[mask, 1], label=class_name, alpha=.6, c=[colors[class_idx]])
    plt.legend()
    plt.title("Patient bag embeddings (Attention-pooled, PCA)")
    pca_plot_path = save_dir / "pca_plot.png"
    plt.savefig(pca_plot_path, bbox_inches="tight")
    plt.close()
    log(f"[5] PCA plot 저장 → {pca_plot_path}")

    log("=" * 55)
    log("DONE: 학습 완료")
    log(f"  model               : {model_path}")
    log(f"  classification_report: {report_path}")
    log(f"  confusion_matrix    : {cm_path}")
    log(f"  auc_metrics         : {auc_path}")
    log(f"  roc_curve_points    : {roc_path}")
    log(f"  roc_curve_plot      : {roc_plot_path}")
    log(f"  pca_coords          : {pca_csv_path}")
    log(f"  pca_plot            : {pca_plot_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
