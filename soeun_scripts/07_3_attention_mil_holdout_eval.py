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
)
from attention_mil_common import AttentionMIL, AttentionMILWrapper

from shared_functions_V2 import predict_labels_and_report_performance


def main():
    parser = argparse.ArgumentParser(description="Attention-MIL holdout 평가")
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--model_name", type=str, default="attention_mil_v1")
    parser.add_argument("--top_k", type=int, default=10,
                        help="attention 해석용 top-k 세포 저장 개수")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gen = "gen3_attention"
    save_dir = MODEL_ROOT / model_gen / "artifacts"

    log("=" * 55)
    log("START: Attention-MIL holdout 평가")
    log(f"  Device : {device}")
    log("=" * 55)

    # ── 모델 로드 ────────────────────────────────────────────────
    model_path = MODEL_ROOT / model_gen / f"{args.model_name}.pt"
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

    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    log(f"[3] holdout balanced accuracy: {bacc:.3f}")
    log(f"[3] holdout F1 (macro)       : {f1m:.3f}")

    report = classification_report(
        y_true, y_pred, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, zero_division=0,
    )
    print(report, flush=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    print(cm, flush=True)

    report_path = save_dir / "holdout_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log(f"  holdout_classification_report 저장 → {report_path}")

    cm_path = save_dir / "holdout_confusion_matrix.txt"
    np.savetxt(cm_path, cm, fmt="%d")
    log(f"  holdout_confusion_matrix 저장 → {cm_path}")

    # ── shared_functions_V2 (팀 공용 포맷과의 일관성 유지) ──────
    log("[4] shared_functions_V2 로 표준 포맷 저장 중...")
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = wrapper,
        holdout_bags = holdout_bags,
        model_gen    = model_gen,
        model_name   = args.model_name,
        output_dir   = str(OUTPUT_DIR),
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
    pca_csv_path = save_dir / "holdout_pca_coords.csv"
    pd.concat([train_df, holdout_df], ignore_index=True).to_csv(pca_csv_path, index=False)
    log(f"[6] PCA 좌표 저장 → {pca_csv_path} (그림은 별도로 그려야 함, matplotlib 미사용)")

    log("=" * 55)
    log("DONE: holdout 평가 완료")
    log(f"  balanced_accuracy : {bacc:.3f}")
    log(f"  f1_macro          : {f1m:.3f}")
    log("=" * 55)


if __name__ == "__main__":
    main()
