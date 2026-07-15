#!/usr/bin/env python3
"""
06_2_cnn_mil_train.py — CNN-MIL(frozen feature + mean pooling) 학습 (CV + 최종 모델)

Attention-MIL(07_2_attention_mil_train.py)과 완전히 동일한 절차를 씁니다.
차이는 pooling 방식뿐입니다: attention 가중합 대신 mean(또는 max) pooling.

최종 모델 학습 방침 (팀원의 06b_train_final_and_eval_holdout.py 와 통일):
  - CV(5-fold)로 "몇 epoch이 적당한지"를 확인
  - 최종 모델은 val 분리 없이 non-holdout 전체 + CV가 알려준 고정 epoch으로 학습
  - holdout은 이 최종 모델로 마지막에 딱 한 번만 평가 (06_3 스크립트)

전제 조건:
  - metadata_for_multiclass.csv 가 PROJECT_DIR 바로 아래 있어야 함
  - python 00_extract_cnn_features.py 로 cache/cnn_features/*.pt 준비되어 있어야 함
    (Attention-MIL과 캐시를 공유합니다 — 특징 추출을 또 할 필요 없음)

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 06_2_cnn_mil_train.py --pooling mean --epochs 60
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    new_run_id, update_latest_symlink,
    plot_multiclass_roc_grid, plot_loss_curves,
)
from cnn_mil_common import (
    CNNMIL, CNNMILWrapper, evaluate, train_model, relabel_bags,
    compute_auc_macro, compute_auc_per_class, compute_roc_curve_points,
)


def make_bag_lookup(all_folders):
    bags = build_bag_objects(all_folders)
    return {b.patient_id: b for b in bags}


def build_model(args, feat_dim):
    return CNNMIL(in_dim=feat_dim, hidden_dim=args.hidden_dim,
                  n_classes=N_CLASSES, pooling=args.pooling, dropout=args.dropout)


def quick_eval(bags, seed, args, device, shuffle_labels=False, epochs=30):
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
    model, _, _, _ = train_model(
        model, train_bags, test_bags, device,
        epochs=epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, verbose=False, use_early_stopping=False,
    )
    _, _, y_true, y_pred, _ = evaluate(model, test_bags, device)
    return balanced_accuracy_score(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="CNN-MIL 학습 (CV + 최종 모델, val 없는 고정 epoch)")
    parser.add_argument("--pooling", choices=["mean", "max", "instance_max"], default="mean")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60,
                        help="CV에서 각 fold를 학습할 때 쓰는 최대 epoch 상한 (early stopping 있음)")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--internal_val_ratio", type=float, default=0.15)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_robustness_seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--run_id", type=str, default=None,
                        help="결과 저장용 run id. 지정 안 하면 현재 시각으로 자동 생성 "
                             "(예: run_20260712_153045). latest 심볼릭 링크가 자동 갱신됨.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="기본값: cnn_mil_v1_{pooling}")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model_gen = "gen2_cnn"
    USER_TAG = "soeun"   # 팀원과 결과 안 겹치게 이 이름으로 하위 폴더 분리
    model_name = args.model_name or f"cnn_mil_v1_{args.pooling}"

    save_dir = MODEL_ROOT / model_gen / USER_TAG / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: CNN-MIL 학습 (CV + 최종 모델)")
    log(f"  Device : {device}")
    log(f"  Pooling: {args.pooling}")
    log(f"  SaveDir: {save_dir}")
    log(f"  RunID  : {run_id}")
    log("=" * 55)

    log(f"[1] {args.metadata_file} 로드 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = get_holdout_folders_from_metadata(meta)
    bag_lookup = make_bag_lookup(meta["folder"].tolist())
    feat_dim = next(iter(bag_lookup.values())).instances.shape[1]
    log(f"[1] 완료 — 전체 {len(meta)}명, holdout {len(holdout_folders)}명 (학습에서 완전히 제외)")

    train_pool_folders = meta.loc[~meta["folder"].isin(holdout_folders), "folder"].tolist()

    log(f"[2] {args.n_folds}-fold CV 시작...")
    fold_reports = []
    fold_best_epochs = []
    fold_train_losses = []
    fold_aucs = []
    agg_y_true, agg_y_pred, agg_y_proba = [], [], []
    fold_data = []

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
        model, _, best_epoch, fold_loss_history = train_model(
            model, fold_train_bags, fold_val_bags, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            class_weights=class_weights, patience=args.patience, verbose=False,
            use_early_stopping=True,
        )
        fold_best_epochs.append(best_epoch)
        fold_train_losses.append(fold_loss_history)

        _, _, y_true, y_pred, y_proba = evaluate(model, fold_test_bags, device)
        agg_y_true.extend(y_true.tolist())
        agg_y_pred.extend(y_pred.tolist())
        agg_y_proba.extend(y_proba.tolist())
        fold_data.append((y_true, y_proba))

        fold_auc = compute_auc_macro(y_true, y_proba, N_CLASSES)
        fold_aucs.append(fold_auc)
        fold_report = classification_report(
            y_true, y_pred, labels=list(range(N_CLASSES)),
            target_names=CLASS_NAMES, zero_division=0,
        )
        fold_reports.append((fold, fold_report))
        bacc = balanced_accuracy_score(y_true, y_pred)
        log(f"  Fold {fold}: n_test={len(fold_test_bags)} | balanced_acc={bacc:.3f} "
            f"| auc_macro={fold_auc:.3f} | best_epoch={best_epoch}")

    agg_y_true  = np.array(agg_y_true)
    agg_y_pred  = np.array(agg_y_pred)
    agg_y_proba = np.array(agg_y_proba)
    agg_bacc = balanced_accuracy_score(agg_y_true, agg_y_pred)
    agg_f1   = f1_score(agg_y_true, agg_y_pred, average="macro", zero_division=0)
    agg_auc  = compute_auc_macro(agg_y_true, agg_y_proba, N_CLASSES)
    agg_auc_per_class = compute_auc_per_class(agg_y_true, agg_y_proba, N_CLASSES)

    final_epochs = int(np.median(fold_best_epochs))
    log(f"[2] CV 완료 — aggregate balanced_acc={agg_bacc:.3f} | f1_macro={agg_f1:.3f} | auc_macro={agg_auc:.3f}")
    log(f"[2] fold별 best_epoch: {fold_best_epochs} → 중앙값 {final_epochs} epoch을 최종 모델에 사용")

    report_path = save_dir / f"classification_report_{args.pooling}.txt"
    with open(report_path, "w") as f:
        f.write("=== CNN-MIL hyperparameters ===\n")
        f.write(json.dumps(vars(args), indent=2) + "\n\n")
        f.write(f"fold별 best_epoch: {fold_best_epochs} -> 최종 모델 epoch = {final_epochs} (중앙값)\n\n")
        for fold, rep in fold_reports:
            f.write(f"=== Fold {fold} ===\n{rep}\n")
        f.write("=== Aggregate (all folds combined) ===\n")
        f.write(classification_report(
            agg_y_true, agg_y_pred, labels=list(range(N_CLASSES)),
            target_names=CLASS_NAMES, zero_division=0,
        ))
    log(f"  classification_report 저장 → {report_path}")

    cm_path = save_dir / f"confusion_matrix_{args.pooling}.txt"
    cm = confusion_matrix(agg_y_true, agg_y_pred, labels=list(range(N_CLASSES)))
    np.savetxt(cm_path, cm, fmt="%d")
    log(f"  confusion_matrix 저장 → {cm_path}")

    auc_path = save_dir / f"auc_metrics_{args.pooling}.json"

    roc_df = compute_roc_curve_points(agg_y_true, agg_y_proba, N_CLASSES, CLASS_NAMES)
    roc_path = save_dir / f"roc_curve_points_{args.pooling}.csv"
    roc_df.to_csv(roc_path, index=False)
    log(f"  ROC curve 원본 좌표(pooled) 저장 → {roc_path}")

    roc_plot_path = save_dir / f"roc_curve_plot_{args.pooling}.png"
    grid_result = plot_multiclass_roc_grid(
        fold_data, CLASS_NAMES, N_CLASSES,
        title=f"ROC curves — CNN-MIL/{args.pooling} (CV, per-fold + mean per subtype)",
        out_path=roc_plot_path,
    )
    auc_path.write_text(json.dumps({
        "per_fold_auc_macro": fold_aucs,
        "aggregate_auc_macro_pooled": agg_auc,
        "aggregate_auc_per_class_pooled": {CLASS_NAMES[c]: v for c, v in agg_auc_per_class.items()},
        "mean_auc_per_class_over_folds": grid_result["mean_auc_per_class"],
        "std_auc_per_class_over_folds": grid_result["std_auc_per_class"],
        "fold_best_epochs": fold_best_epochs,
        "final_model_epochs": final_epochs,
    }, indent=2))
    log(f"  ROC curve plot(2x3) 저장 → {roc_plot_path}")
    log(f"  AUC 결과(pooled + fold별 mean±std 둘 다) 갱신 → {auc_path}")


    log(f"[3] 최종 모델 학습 시작 (non-holdout {len(train_pool_folders)}명 전체, "
        f"val 분리 없음, 고정 {final_epochs} epoch)...")
    train_pool_bags = [bag_lookup[f] for f in train_pool_folders]
    print_distribution("최종 Train (전체, val 분리 없음)",
                        np.array([b.true_label for b in train_pool_bags]))

    class_weights = compute_class_weights(np.array([b.true_label for b in train_pool_bags]))
    model = build_model(args, feat_dim)
    model, _, _, final_train_loss = train_model(
        model, train_pool_bags, None, device,
        epochs=final_epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=class_weights, verbose=True, use_early_stopping=False,
    )
    log(f"[3] 학습 완료 (고정 {final_epochs} epoch, val 없음)")

    loss_plot_path = save_dir / f"loss_curve_plot_{args.pooling}.png"
    plot_loss_curves(
        fold_train_losses, fold_best_epochs, final_train_loss,
        title=f"Train loss — CNN-MIL/{args.pooling} (CV folds + 최종 모델)",
        out_path=loss_plot_path,
    )
    log(f"  loss curve plot 저장 → {loss_plot_path}")

    model_dir = USER_TAG_DIR / run_id
    model_path = model_dir / f"{model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": feat_dim, "hidden_dim": args.hidden_dim,
        "n_classes": N_CLASSES, "pooling": args.pooling,
        "dropout": args.dropout, "classes": CLASS_NAMES,
        "final_epochs": final_epochs,
    }, model_path)
    log(f"  모델 저장 완료 → {model_path}")

    log(f"[4] robustness 체크 시작 ({args.n_robustness_seeds} seeds, real vs shuffled labels)...")
    real_scores = [quick_eval(train_pool_bags, s, args, device, shuffle_labels=False)
                   for s in range(args.n_robustness_seeds)]
    shuf_scores = [quick_eval(train_pool_bags, s, args, device, shuffle_labels=True)
                   for s in range(args.n_robustness_seeds)]

    log(f"  real labels : {np.round(real_scores, 3).tolist()} mean {np.mean(real_scores):.3f}")
    log(f"  shuffled    : {np.round(shuf_scores, 3).tolist()} mean {np.mean(shuf_scores):.3f}")
    if np.mean(shuf_scores) > 0.35:
        log("  [WARN] 셔플 라벨인데도 balanced accuracy가 높습니다 — 데이터 누수 의심, 확인 필요")

    robustness_path = save_dir / f"robustness_check_{args.pooling}.json"
    robustness_path.write_text(json.dumps({
        "real_scores": real_scores, "real_mean": float(np.mean(real_scores)),
        "shuffled_scores": shuf_scores, "shuffled_mean": float(np.mean(shuf_scores)),
    }, indent=2))
    log(f"  robustness 결과 저장 → {robustness_path}")

    if args.pooling in ("mean", "max"):
        log("[5] PCA 시각화 준비 중...")
        wrapper = CNNMILWrapper(model, device)
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

        np.save(save_dir / f"X_bagvec_{args.pooling}.npy", X_bagvec)
        np.save(save_dir / f"groups_bagvec_{args.pooling}.npy", groups)
        np.save(save_dir / f"labels_bagvec_{args.pooling}.npy", labels)

        scaler = StandardScaler().fit(X_bagvec)
        pca = PCA(n_components=2).fit(scaler.transform(X_bagvec))
        Z = pca.transform(scaler.transform(X_bagvec))

        import pandas as pd
        pca_df = pd.DataFrame({
            "patient_id": groups,
            "true_subtype": [CLASS_NAMES[l] for l in labels],
            "pc1": Z[:, 0], "pc2": Z[:, 1],
        })
        pca_csv_path = save_dir / f"pca_coords_{args.pooling}.csv"
        pca_df.to_csv(pca_csv_path, index=False)

        colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
        plt.figure()
        for class_idx, class_name in enumerate(CLASS_NAMES):
            mask = labels == class_idx
            plt.scatter(Z[mask, 0], Z[mask, 1], label=class_name, alpha=.6, c=[colors[class_idx]])
        plt.legend()
        plt.title(f"Patient bag embeddings ({args.pooling} pooling, PCA)")
        pca_plot_path = save_dir / f"pca_plot_{args.pooling}.png"
        plt.savefig(pca_plot_path, bbox_inches="tight")
        plt.close()
        log(f"[5] PCA plot 저장 → {pca_plot_path}")
    else:
        log("[5] instance_max pooling은 bag 벡터가 없어 PCA를 건너뜁니다.")

    update_latest_symlink(USER_TAG_DIR, run_id)

    log("=" * 55)
    log("DONE: 학습 완료")
    log(f"  model               : {model_path}")
    log(f"  final_model_epochs  : {final_epochs} (CV fold별 best_epoch 중앙값)")
    log(f"  classification_report: {report_path}")
    log(f"  confusion_matrix    : {cm_path}")
    log(f"  auc_metrics         : {auc_path}")
    log(f"  roc_curve_plot      : {roc_plot_path}")
    log(f"  loss_curve_plot     : {loss_plot_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
