#!/usr/bin/env python3
"""
07_3_attention_mil_holdout_eval.py — 저장된 Attention-MIL 모델을 holdout에 평가

팀원의 j_SVM_multiclass_holdout_eval.py 와 동일한 산출물 형태:
  - holdout_classification_report.txt
  - holdout_confusion_matrix.txt
  - holdout_pca_plot.png   (train 임베딩 위에 holdout을 'x'로 겹쳐 그림,
                             PCA/scaler는 train 벡터에 fit — holdout 단독으로
                             fit하면 노이즈가 커서 train과 같은 축을 재사용)

전제 조건:
  - 07_2_attention_mil_train.py 를 먼저 실행해서
      models/gen3_attention/attention_mil_v1.pt
      models/gen3_attention/artifacts/X_bagvec.npy / groups_bagvec.npy / labels_bagvec.npy
    가 준비되어 있어야 함

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_3_attention_mil_holdout_eval.py
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
    log, build_bag_objects, get_instance_filenames,
    load_metadata, get_holdout_folders_from_metadata,
    plot_multiclass_roc_grid,
)
from attention_mil_common import (
    AttentionMIL, AttentionMILWrapper, compute_auc_macro, compute_auc_per_class,
    compute_roc_curve_points,
)

from shared_functions_V2 import predict_labels_and_report_performance


def main():
    parser = argparse.ArgumentParser(description="Attention-MIL holdout 평가")
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--run_id", type=str, default="latest",
                        help="어느 학습 실행 결과를 쓸지. 기본값 latest는 "
                             "가장 최근 학습(07_2/07_6/06_2)이 갱신한 심볼릭 링크를 따라감.")
    parser.add_argument("--model_name", type=str, default="attention_mil_v1")
    parser.add_argument("--top_k", type=int, default=10,
                        help="attention 해석용 top-k 세포 저장 개수")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gen = "gen3_attention"
    USER_TAG = "soeun"
    USER_TAG_DIR = MODEL_ROOT / model_gen / USER_TAG
    run_id = args.run_id

    save_dir = USER_TAG_DIR / run_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: Attention-MIL holdout 평가")
    log(f"  Device : {device}")
    log("=" * 55)

    # ── 모델 로드 ────────────────────────────────────────────────
    model_path = USER_TAG_DIR / run_id / f"{args.model_name}.pt"
    log(f"[1] 모델 로드 중 → {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = AttentionMIL(
        in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], attn_dim=ckpt["attn_dim"],
        n_classes=ckpt["n_classes"], dropout=ckpt["dropout"], gated=ckpt["gated"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    wrapper = AttentionMILWrapper(model, device)
    log("[1] 완료")

    # ── holdout 환자 목록 + 임베딩 로드 ──────────────────────────
    log(f"[2] {args.metadata_file} 로드 및 holdout bag 구성 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = sorted(get_holdout_folders_from_metadata(meta))
    holdout_bags = build_bag_objects(holdout_folders)
    log(f"[2] 완료 — holdout {len(holdout_bags)}명")

    # ── 예측 ────────────────────────────────────────────────────
    log("[3] holdout 예측 중...")
    y_true = np.array([b.true_label for b in holdout_bags])
    y_pred = np.array([wrapper.predict_bag(b.instances)["pred_label"] for b in holdout_bags])
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

    auc_path = save_dir / "holdout_auc_metrics_attention.json"

    roc_df = compute_roc_curve_points(y_true, y_proba, N_CLASSES, CLASS_NAMES)
    roc_path = save_dir / "holdout_roc_curve_points_attention.csv"
    roc_df.to_csv(roc_path, index=False)
    log(f"  holdout ROC curve 원본 좌표 저장 → {roc_path}")

    roc_plot_path = save_dir / "holdout_roc_curve_plot_attention.png"
    grid_result = plot_multiclass_roc_grid(
        [(y_true, y_proba)], CLASS_NAMES, N_CLASSES,
        title="ROC curves — Attention-MIL (holdout)",
        out_path=roc_plot_path,
    )
    auc_path.write_text(json.dumps({
        "auc_macro": auc_macro,
        "auc_per_class": {CLASS_NAMES[c]: v for c, v in auc_per_class.items()},
        "auc_per_class_grid": grid_result["mean_auc_per_class"],  # 위와 동일해야 함 (교차검산용)
    }, indent=2))
    log(f"  holdout AUC 결과 저장 → {auc_path}")
    log(f"  holdout ROC curve plot(2x3) 저장 → {roc_plot_path}")

    report = classification_report(
        y_true, y_pred, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, zero_division=0,
    )
    print(report, flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print(cm, flush=True)

    report_path = save_dir / "holdout_classification_report_attention.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log(f"  holdout_classification_report 저장 → {report_path}")

    cm_path = save_dir / "holdout_confusion_matrix_attention.txt"
    np.savetxt(cm_path, cm, fmt="%d")
    log(f"  holdout_confusion_matrix 저장 → {cm_path}")

    # ── shared_functions_V2 (팀 공용 포맷과의 일관성 유지) ──────
    log("[4] shared_functions_V2 로 표준 포맷 저장 중...")
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = wrapper,
        holdout_bags = holdout_bags,
        model_gen    = model_gen,
        model_name   = args.model_name,
        output_dir   = str(OUTPUT_DIR / USER_TAG / run_id),
    )
    log("[4] 완료")
    print(metrics_df.to_string(index=False), flush=True)

    # ── attention top-k 세포 저장 (해석용) ─────────────────────
    log("[5] top-attention 세포 저장 중...")
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
                "patient_id": bag.patient_id,
                "true_label": CLASS_NAMES[bag.true_label],
                "pred_label": CLASS_NAMES[result["pred_label"]],
                "rank": rank, "filename": filenames[i],
                "attn_weight": float(attn[i]),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(save_dir / "attention_top_cells_holdout.csv", index=False)
    log(f"[5] 완료 → {save_dir / 'attention_top_cells_holdout.csv'}")

    # ── PCA overlay plot (train 벡터에 fit, holdout을 겹쳐 그림) ─
    log("[6] PCA overlay plot 생성 중...")
    X_train = np.load(save_dir / "X_bagvec.npy")
    y_train = np.load(save_dir / "labels_bagvec.npy")

    X_holdout = np.array([wrapper.get_bag_vector(b.instances) for b in holdout_bags])

    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=2).fit(scaler.transform(X_train))
    Z_train   = pca.transform(scaler.transform(X_train))
    Z_holdout = pca.transform(scaler.transform(X_holdout))

    train_groups = np.load(save_dir / "groups_bagvec.npy")
    import pandas as pd
    train_df = pd.DataFrame({
        "patient_id": train_groups,
        "true_subtype": [CLASS_NAMES[l] for l in y_train],
        "split": "train",
        "pc1": Z_train[:, 0], "pc2": Z_train[:, 1],
    })
    holdout_df = pd.DataFrame({
        "patient_id": [b.patient_id for b in holdout_bags],
        "true_subtype": [CLASS_NAMES[l] for l in y_true],
        "split": "holdout",
        "pc1": Z_holdout[:, 0], "pc2": Z_holdout[:, 1],
    })
    pca_csv_path = save_dir / "holdout_pca_coords_attention.csv"
    pd.concat([train_df, holdout_df], ignore_index=True).to_csv(pca_csv_path, index=False)
    log(f"[6] PCA 좌표 저장 → {pca_csv_path}")

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
    plt.title("Holdout patient bag embeddings (PCA, fit on training set)")
    pca_plot_path = save_dir / "holdout_pca_plot_attention.png"
    plt.savefig(pca_plot_path, bbox_inches="tight")
    plt.close()
    log(f"[6] PCA plot 저장 → {pca_plot_path}")

    log("=" * 55)
    log("DONE: holdout 평가 완료")
    log(f"  balanced_accuracy : {bacc:.3f}")
    log(f"  f1_macro          : {f1m:.3f}")
    log(f"  auc_macro         : {auc_macro:.3f}")
    log(f"  roc_curve_plot    : {roc_plot_path}")
    log(f"  pca_plot          : {pca_plot_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
