#!/usr/bin/env python3
"""
06_3_cnn_mil_holdout_eval.py — 저장된 CNN-MIL 모델을 holdout에 평가

07_3_attention_mil_holdout_eval.py 와 동일한 절차/산출물 형태입니다.
재학습 없이 06_2에서 저장한 최종 모델을 불러와 forward만 수행합니다.

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 06_3_cnn_mil_holdout_eval.py --pooling mean
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, classification_report,
)

from mil_common import (
    PROJECT_DIR, MODEL_ROOT, OUTPUT_DIR, CLASS_NAMES, N_CLASSES,
    log, build_bag_objects,
    load_metadata, get_holdout_folders_from_metadata,
    plot_multiclass_roc_grid,
)
from cnn_mil_common import (
    CNNMIL, CNNMILWrapper, compute_auc_macro, compute_auc_per_class,
    compute_roc_curve_points,
)

from shared_functions_V2 import predict_labels_and_report_performance


def main():
    parser = argparse.ArgumentParser(description="CNN-MIL holdout 평가")
    parser.add_argument("--pooling", choices=["mean", "max", "instance_max"], default="mean")
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--run_id", type=str, default="latest",
                        help="어느 학습 실행 결과를 쓸지. 기본값 latest는 "
                             "가장 최근 학습(07_2/07_6/06_2)이 갱신한 심볼릭 링크를 따라감.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="기본값: cnn_mil_v1_{pooling}")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gen = "gen2_cnn"
    USER_TAG = "soeun"
    USER_TAG_DIR = MODEL_ROOT / model_gen / USER_TAG
    run_id = args.run_id
    model_name = args.model_name or f"cnn_mil_v1_{args.pooling}"
    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: CNN-MIL holdout 평가")
    log(f"  Device : {device}")
    log(f"  Pooling: {args.pooling}")
    log("=" * 55)

    model_path = USER_TAG_DIR / run_id / f"{model_name}.pt"
    log(f"[1] 모델 로드 중 → {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = CNNMIL(
        in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"],
        n_classes=ckpt["n_classes"], pooling=ckpt["pooling"], dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    wrapper = CNNMILWrapper(model, device)
    log("[1] 완료")

    log(f"[2] {args.metadata_file} 로드 및 holdout bag 구성 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = sorted(get_holdout_folders_from_metadata(meta))
    holdout_bags = build_bag_objects(holdout_folders)
    log(f"[2] 완료 — holdout {len(holdout_bags)}명")

    log("[3] holdout 예측 중...")
    y_true  = np.array([b.true_label for b in holdout_bags])
    y_pred  = np.array([wrapper.predict_bag(b.instances)["pred_label"] for b in holdout_bags])
    y_proba = np.array([wrapper.predict_bag_proba(b.instances) for b in holdout_bags])

    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc_macro = compute_auc_macro(y_true, y_proba, N_CLASSES)
    auc_per_class = compute_auc_per_class(y_true, y_proba, N_CLASSES)
    log(f"[3] holdout balanced accuracy: {bacc:.3f}")
    log(f"[3] holdout F1 (macro)       : {f1m:.3f}")
    log(f"[3] holdout AUC (macro, OvR) : {auc_macro:.3f}")
    for c, name in enumerate(CLASS_NAMES):
        log(f"       AUC[{name:<16}]: {auc_per_class[c]:.3f}")

    report = classification_report(
        y_true, y_pred, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, zero_division=0,
    )
    print(report, flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print(cm, flush=True)

    report_path = save_dir / f"holdout_classification_report_{args.pooling}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log(f"  holdout_classification_report 저장 → {report_path}")

    cm_path = save_dir / f"holdout_confusion_matrix_{args.pooling}.txt"
    np.savetxt(cm_path, cm, fmt="%d")
    log(f"  holdout_confusion_matrix 저장 → {cm_path}")

    auc_path = save_dir / f"holdout_auc_metrics_{args.pooling}.json"

    roc_df = compute_roc_curve_points(y_true, y_proba, N_CLASSES, CLASS_NAMES)
    roc_path = save_dir / f"holdout_roc_curve_points_{args.pooling}.csv"
    roc_df.to_csv(roc_path, index=False)

    roc_plot_path = save_dir / f"holdout_roc_curve_plot_{args.pooling}.png"
    grid_result = plot_multiclass_roc_grid(
        [(y_true, y_proba)], CLASS_NAMES, N_CLASSES,
        title=f"ROC curves — CNN-MIL/{args.pooling} (holdout)",
        out_path=roc_plot_path,
    )
    auc_path.write_text(json.dumps({
        "auc_macro": auc_macro,
        "auc_per_class": {CLASS_NAMES[c]: v for c, v in auc_per_class.items()},
        "auc_per_class_grid": grid_result["mean_auc_per_class"],
    }, indent=2))
    log(f"  holdout AUC 결과 저장 → {auc_path}")
    log(f"  holdout ROC curve plot(2x3) 저장 → {roc_plot_path}")

    log("[4] shared_functions_V2 로 표준 포맷 저장 중...")
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = wrapper,
        holdout_bags = holdout_bags,
        model_gen    = model_gen,
        model_name   = model_name,
        output_dir   = str(OUTPUT_DIR / USER_TAG / run_id),
    )
    log("[4] 완료")
    print(metrics_df.to_string(index=False), flush=True)

    if args.pooling in ("mean", "max"):
        log("[5] PCA overlay plot 생성 중...")
        X_train_path = save_dir / f"X_bagvec_{args.pooling}.npy"
        if X_train_path.exists():
            X_train = np.load(X_train_path)
            y_train = np.load(save_dir / f"labels_bagvec_{args.pooling}.npy")
            X_holdout = np.array([wrapper.get_bag_vector(b.instances) for b in holdout_bags])

            scaler = StandardScaler().fit(X_train)
            pca = PCA(n_components=2).fit(scaler.transform(X_train))
            Z_train   = pca.transform(scaler.transform(X_train))
            Z_holdout = pca.transform(scaler.transform(X_holdout))

            colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
            plt.figure()
            for class_idx, class_name in enumerate(CLASS_NAMES):
                tm = y_train == class_idx
                hm = y_true == class_idx
                plt.scatter(Z_train[tm, 0], Z_train[tm, 1],
                            label=f"train: {class_name}", alpha=.3, c=[colors[class_idx]])
                plt.scatter(Z_holdout[hm, 0], Z_holdout[hm, 1],
                            label=f"holdout: {class_name}", marker="x", s=100, c=[colors[class_idx]])
            plt.legend(fontsize=7)
            plt.title(f"Holdout bag embeddings ({args.pooling} pooling, PCA fit on training set)")
            pca_plot_path = save_dir / f"holdout_pca_plot_{args.pooling}.png"
            plt.savefig(pca_plot_path, bbox_inches="tight")
            plt.close()
            log(f"[5] PCA plot 저장 → {pca_plot_path}")
        else:
            log(f"[5] {X_train_path} 없음 — 06_2_cnn_mil_train.py를 먼저 실행하세요. PCA 건너뜀.")

    log("=" * 55)
    log("DONE: holdout 평가 완료")
    log(f"  balanced_accuracy : {bacc:.3f}")
    log(f"  f1_macro          : {f1m:.3f}")
    log(f"  auc_macro         : {auc_macro:.3f}")
    log("=" * 55)


if __name__ == "__main__":
    main()
