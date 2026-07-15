#!/usr/bin/env python3
"""
08_1_attention_finetune_train.py — Attention-MIL + ResNet50 fine-tuning 학습

기존 07_2(frozen feature)와 절차는 같습니다: 5-fold CV로 적정 epoch 확인 →
그 중앙값으로 non-holdout 전체를 학습하는 최종 모델. 차이는 backbone을
얼려두지 않고 layer4를 fine-tuning한다는 것뿐입니다 (그래서 훨씬 오래 걸립니다).

★ 미리 각오하실 것: 이미지 원본을 매 epoch CNN에 통과시키므로,
   frozen feature 버전보다 수십~수백 배 오래 걸립니다.
   (팀원 CNN 로그 기준 fold 하나에 몇 분~십수 분)

전제 조건:
  - metadata_for_multiclass.csv 가 PROJECT_DIR 바로 아래 있어야 함
  - organized_data/{환자폴더}/*.tif 원본 이미지가 그대로 있어야 함
    (cache/cnn_features/*.pt 는 필요 없음 — 이미지 원본을 직접 씁니다)

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 08_1_attention_finetune_train.py --max_epochs 100 --unfreeze_from layer4
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, classification_report,
)

from mil_common import (
    PROJECT_DIR, MODEL_ROOT, CLASS_NAMES, N_CLASSES, ORGANIZED_DIR,
    log, print_distribution, compute_class_weights, parse_label_from_folder,
    load_metadata, get_holdout_folders_from_metadata, get_fold_split,
    plot_multiclass_roc_grid, plot_loss_curves,
    new_run_id, update_latest_symlink,
)
from attention_mil_common import compute_auc_macro, compute_auc_per_class, compute_roc_curve_points
from attention_finetune_common import (
    AttentionMILFineTune, ClassWiseAttentionMILFineTune,
    train_model_finetune, evaluate_patients,
)


def build_model(args):
    if args.pooling_type == "gated":
        return AttentionMILFineTune(
            hidden_dim=args.hidden_dim, attn_dim=args.attn_dim, n_classes=N_CLASSES,
            dropout=args.dropout, gated=args.gated, unfreeze_from=args.unfreeze_from,
        )
    else:  # "classwise"
        return ClassWiseAttentionMILFineTune(
            hidden_dim=args.hidden_dim, attn_dim=args.attn_dim, n_classes=N_CLASSES,
            dropout=args.dropout, unfreeze_from=args.unfreeze_from,
        )


def main():
    parser = argparse.ArgumentParser(description="Attention-MIL + ResNet50 fine-tuning 학습")
    parser.add_argument("--pooling_type", choices=["gated", "classwise"], default="gated",
                        help="gated: Ilse et al. 공유 attention / classwise: SCEMILA 방식 클래스별 독립 attention")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--gated", action="store_true", default=True)
    parser.add_argument("--no_gated", dest="gated", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze_from", default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4", "all", "none"])
    parser.add_argument("--instances_per_step", type=int, default=32)
    parser.add_argument("--lr_head", type=float, default=1e-2)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="CV에서 각 fold를 학습할 때 쓰는 최대 epoch 상한 (early stopping 있음)")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--internal_val_ratio", type=float, default=0.15)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--image_ext", default=".tif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None,
                        help="기본값: attention_finetune_v1_{pooling_type}")
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = f"attention_finetune_v1_{args.pooling_type}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model_gen = f"gen4_attention_finetune_{args.pooling_type}"
    USER_TAG = "soeun"
    run_id = args.run_id or new_run_id()
    USER_TAG_DIR = MODEL_ROOT / model_gen / USER_TAG

    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: Attention-MIL + ResNet50 fine-tuning 학습")
    log(f"  Device       : {device}")
    log(f"  Unfreeze from: {args.unfreeze_from}")
    log(f"  RunID        : {run_id}")
    log(f"  SaveDir      : {save_dir}")
    log("  [경고] 이미지 원본을 매 epoch CNN에 통과시켜서 매우 오래 걸립니다.")
    log("=" * 55)

    log(f"[1] {args.metadata_file} 로드 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = get_holdout_folders_from_metadata(meta)
    all_folders = meta["folder"].tolist()
    labels = {f: parse_label_from_folder(f) for f in all_folders}
    train_pool_folders = [f for f in all_folders if f not in holdout_folders]
    log(f"[1] 완료 — 전체 {len(all_folders)}명, holdout {len(holdout_folders)}명 (학습에서 완전히 제외)")

    # ── 5-fold CV ──────────────────────────────────────────────
    log(f"[2] {args.n_folds}-fold CV 시작...")
    fold_reports, fold_best_epochs, fold_aucs = [], [], []
    agg_y_true, agg_y_pred, agg_y_proba = [], [], []
    fold_data, fold_train_losses = [], []

    for fold in range(1, args.n_folds + 1):
        train_folders, test_folders = get_fold_split(meta, fold)
        y_train_all = np.array([parse_label_from_folder(f) for f in train_folders])
        tr_idx, val_idx = train_test_split(
            np.arange(len(train_folders)), test_size=args.internal_val_ratio,
            stratify=y_train_all, random_state=args.seed,
        )
        fold_train_ids = [train_folders[i] for i in tr_idx]
        fold_val_ids   = [train_folders[i] for i in val_idx]

        class_weights = compute_class_weights(np.array([labels[f] for f in fold_train_ids]))
        model = build_model(args)
        model, _, best_epoch, fold_loss_history = train_model_finetune(
            model, ORGANIZED_DIR, fold_train_ids, fold_val_ids, labels, args.image_ext,
            device, max_epochs=args.max_epochs, lr_head=args.lr_head, lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay, class_weights=class_weights,
            instances_per_step=args.instances_per_step, patience=args.patience,
            seed=args.seed, verbose=False, use_early_stopping=True,
        )
        fold_best_epochs.append(best_epoch)
        fold_train_losses.append(fold_loss_history)

        y_true, y_pred, y_proba = evaluate_patients(model, ORGANIZED_DIR, test_folders, labels, args.image_ext, device)
        agg_y_true.extend(y_true.tolist()); agg_y_pred.extend(y_pred.tolist()); agg_y_proba.extend(y_proba.tolist())
        fold_data.append((y_true, y_proba))

        fold_auc = compute_auc_macro(y_true, y_proba, N_CLASSES)
        fold_aucs.append(fold_auc)
        fold_reports.append((fold, classification_report(
            y_true, y_pred, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, zero_division=0,
        )))
        bacc = balanced_accuracy_score(y_true, y_pred)
        log(f"  Fold {fold}: n_test={len(test_folders)} | balanced_acc={bacc:.3f} "
            f"| auc_macro={fold_auc:.3f} | best_epoch={best_epoch}")

    agg_y_true, agg_y_pred, agg_y_proba = np.array(agg_y_true), np.array(agg_y_pred), np.array(agg_y_proba)
    agg_bacc = balanced_accuracy_score(agg_y_true, agg_y_pred)
    agg_f1 = f1_score(agg_y_true, agg_y_pred, average="macro", zero_division=0)
    agg_auc = compute_auc_macro(agg_y_true, agg_y_proba, N_CLASSES)
    agg_auc_per_class = compute_auc_per_class(agg_y_true, agg_y_proba, N_CLASSES)
    final_epochs = int(np.median(fold_best_epochs))
    log(f"[2] CV 완료 — balanced_acc={agg_bacc:.3f} | f1_macro={agg_f1:.3f} | auc_macro={agg_auc:.3f}")
    log(f"[2] fold별 best_epoch: {fold_best_epochs} → 중앙값 {final_epochs} epoch을 최종 모델에 사용")

    report_path = save_dir / f"classification_report_finetune_{args.pooling_type}.txt"
    with open(report_path, "w") as f:
        f.write("=== Attention-MIL(fine-tune) hyperparameters ===\n")
        f.write(json.dumps(vars(args), indent=2) + "\n\n")
        f.write(f"fold별 best_epoch: {fold_best_epochs} -> 최종 모델 epoch = {final_epochs}\n\n")
        for fold, rep in fold_reports:
            f.write(f"=== Fold {fold} ===\n{rep}\n")
        f.write("=== Aggregate ===\n")
        f.write(classification_report(agg_y_true, agg_y_pred, labels=list(range(N_CLASSES)),
                                       target_names=CLASS_NAMES, zero_division=0))
    log(f"  classification_report 저장 → {report_path}")

    cm_path = save_dir / f"confusion_matrix_finetune_{args.pooling_type}.txt"
    np.savetxt(cm_path, confusion_matrix(agg_y_true, agg_y_pred, labels=list(range(N_CLASSES))), fmt="%d")
    log(f"  confusion_matrix 저장 → {cm_path}")

    roc_path = save_dir / f"roc_curve_points_finetune_{args.pooling_type}.csv"
    compute_roc_curve_points(agg_y_true, agg_y_proba, N_CLASSES, CLASS_NAMES).to_csv(roc_path, index=False)

    roc_plot_path = save_dir / f"roc_curve_plot_finetune_{args.pooling_type}.png"
    grid_result = plot_multiclass_roc_grid(
        fold_data, CLASS_NAMES, N_CLASSES,
        title="ROC curves — Attention-MIL fine-tune (CV, per-fold + mean per subtype)",
        out_path=roc_plot_path,
    )
    auc_path = save_dir / f"auc_metrics_finetune_{args.pooling_type}.json"
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

    # ── 최종 모델: val 분리 없이 non-holdout 전체, 고정 epoch ──
    log(f"[3] 최종 모델 학습 시작 (non-holdout {len(train_pool_folders)}명 전체, "
        f"val 분리 없음, 고정 {final_epochs} epoch)...")
    print_distribution("최종 Train (전체, val 분리 없음)",
                        np.array([labels[f] for f in train_pool_folders]))

    class_weights = compute_class_weights(np.array([labels[f] for f in train_pool_folders]))
    model = build_model(args)
    model, _, _, final_train_loss = train_model_finetune(
        model, ORGANIZED_DIR, train_pool_folders, [], labels, args.image_ext,
        device, max_epochs=final_epochs, lr_head=args.lr_head, lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay, class_weights=class_weights,
        instances_per_step=args.instances_per_step, seed=args.seed,
        verbose=True, use_early_stopping=False,
    )
    log(f"[3] 학습 완료 (고정 {final_epochs} epoch, val 없음)")

    loss_plot_path = save_dir / f"loss_curve_plot_finetune_{args.pooling_type}.png"
    plot_loss_curves(
        fold_train_losses, fold_best_epochs, final_train_loss,
        title="Train loss — Attention-MIL fine-tune (CV folds + final model)",
        out_path=loss_plot_path,
    )
    log(f"  loss curve plot 저장 → {loss_plot_path}")

    model_dir = USER_TAG_DIR / run_id
    model_path = model_dir / f"{args.model_name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "pooling_type": args.pooling_type,
        "hidden_dim": args.hidden_dim, "attn_dim": args.attn_dim, "gated": args.gated,
        "n_classes": N_CLASSES, "dropout": args.dropout, "unfreeze_from": args.unfreeze_from,
        "classes": CLASS_NAMES, "final_epochs": final_epochs,
    }, model_path)
    log(f"  모델 저장 완료 → {model_path}")

    update_latest_symlink(USER_TAG_DIR, run_id)

    log("=" * 55)
    log("DONE: 학습 완료")
    log(f"  model               : {model_path}")
    log(f"  final_model_epochs  : {final_epochs}")
    log(f"  classification_report: {report_path}")
    log(f"  confusion_matrix    : {cm_path}")
    log(f"  auc_metrics         : {auc_path}")
    log(f"  roc_curve_plot      : {roc_plot_path}")
    log(f"  loss_curve_plot     : {loss_plot_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
