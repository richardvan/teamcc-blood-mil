#!/usr/bin/env python3
"""
08_2_attention_finetune_holdout_eval.py — 저장된 fine-tune Attention-MIL 모델을 holdout에 평가

재학습 없이 08_1이 저장한 모델을 불러와 holdout 28명에 대해서만 평가합니다.
그래도 이미지 원본을 backbone에 통과시켜야 해서, frozen feature 버전보다는 느립니다
(다만 holdout은 28명뿐이라 학습 자체보다는 훨씬 빠르게 끝납니다).

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 08_2_attention_finetune_holdout_eval.py
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, classification_report,
)

from mil_common import (
    PROJECT_DIR, MODEL_ROOT, CLASS_NAMES, N_CLASSES, ORGANIZED_DIR,
    log, parse_label_from_folder,
    load_metadata, get_holdout_folders_from_metadata,
    plot_multiclass_roc_grid,
)
from attention_mil_common import compute_auc_macro, compute_auc_per_class, compute_roc_curve_points
from attention_finetune_common import (
    AttentionMILFineTune, ClassWiseAttentionMILFineTune, evaluate_patients, predict_patient,
)


def main():
    parser = argparse.ArgumentParser(description="Attention-MIL fine-tune holdout 평가")
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--run_id", type=str, default="latest")
    parser.add_argument("--pooling_type", choices=["gated", "classwise"], default="gated")
    parser.add_argument("--model_name", type=str, default=None,
                        help="기본값: attention_finetune_v1_{pooling_type}")
    parser.add_argument("--image_ext", default=".tif")
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = f"attention_finetune_v1_{args.pooling_type}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gen = f"gen4_attention_finetune_{args.pooling_type}"
    USER_TAG = "soeun"
    USER_TAG_DIR = MODEL_ROOT / model_gen / USER_TAG
    run_id = args.run_id
    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: Attention-MIL fine-tune holdout 평가")
    log(f"  Device : {device}")
    log("=" * 55)

    model_path = USER_TAG_DIR / run_id / f"{args.model_name}.pt"
    log(f"[1] 모델 로드 중 → {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    pooling_type = ckpt.get("pooling_type", args.pooling_type)
    if pooling_type == "gated":
        model = AttentionMILFineTune(
            hidden_dim=ckpt["hidden_dim"], attn_dim=ckpt["attn_dim"], n_classes=ckpt["n_classes"],
            dropout=ckpt["dropout"], gated=ckpt["gated"], unfreeze_from=ckpt["unfreeze_from"],
        )
    else:
        model = ClassWiseAttentionMILFineTune(
            hidden_dim=ckpt["hidden_dim"], attn_dim=ckpt["attn_dim"], n_classes=ckpt["n_classes"],
            dropout=ckpt["dropout"], unfreeze_from=ckpt["unfreeze_from"],
        )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    log("[1] 완료")

    log(f"[2] {args.metadata_file} 로드 및 holdout 목록 구성 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = sorted(get_holdout_folders_from_metadata(meta))
    labels = {f: parse_label_from_folder(f) for f in holdout_folders}
    log(f"[2] 완료 — holdout {len(holdout_folders)}명")

    log("[3] holdout 예측 중 (환자별 전체 이미지 사용, 시간 다소 소요)...")
    y_true, y_pred, y_proba = evaluate_patients(
        model, ORGANIZED_DIR, holdout_folders, labels, args.image_ext, device,
    )

    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc_macro = compute_auc_macro(y_true, y_proba, N_CLASSES)
    auc_per_class = compute_auc_per_class(y_true, y_proba, N_CLASSES)
    log(f"[3] holdout balanced accuracy: {bacc:.3f}")
    log(f"[3] holdout F1 (macro)       : {f1m:.3f}")
    log(f"[3] holdout AUC (macro, OvR) : {auc_macro:.3f}")
    for c, name in enumerate(CLASS_NAMES):
        log(f"       AUC[{name:<16}]: {auc_per_class[c]:.3f}")

    report = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)),
                                    target_names=CLASS_NAMES, zero_division=0)
    print(report, flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print(cm, flush=True)

    (save_dir / f"holdout_classification_report_finetune_{args.pooling_type}.txt").write_text(report)
    np.savetxt(save_dir / f"holdout_confusion_matrix_finetune_{args.pooling_type}.txt", cm, fmt="%d")

    roc_df = compute_roc_curve_points(y_true, y_proba, N_CLASSES, CLASS_NAMES)
    roc_df.to_csv(save_dir / f"holdout_roc_curve_points_finetune_{args.pooling_type}.csv", index=False)

    roc_plot_path = save_dir / f"holdout_roc_curve_plot_finetune_{args.pooling_type}.png"
    grid_result = plot_multiclass_roc_grid(
        [(y_true, y_proba)], CLASS_NAMES, N_CLASSES,
        title="ROC curves — Attention-MIL fine-tune (holdout)",
        out_path=roc_plot_path,
    )
    auc_path = save_dir / f"holdout_auc_metrics_finetune_{args.pooling_type}.json"
    auc_path.write_text(json.dumps({
        "auc_macro": auc_macro,
        "auc_per_class": {CLASS_NAMES[c]: v for c, v in auc_per_class.items()},
        "auc_per_class_grid": grid_result["mean_auc_per_class"],
    }, indent=2))

    log(f"  holdout_classification_report / confusion_matrix / AUC / ROC 저장 완료 → {save_dir}")
    log("=" * 55)
    log("DONE: holdout 평가 완료")
    log(f"  balanced_accuracy : {bacc:.3f}")
    log(f"  f1_macro          : {f1m:.3f}")
    log(f"  auc_macro         : {auc_macro:.3f}")
    log("=" * 55)


if __name__ == "__main__":
    main()
