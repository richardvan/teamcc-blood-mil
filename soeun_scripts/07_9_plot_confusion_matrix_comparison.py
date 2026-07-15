#!/usr/bin/env python3
"""
07_9_plot_confusion_matrix_comparison.py — CV / Holdout confusion matrix를 나란히 그리기

이미 저장된 confusion_matrix_{tag}.txt / holdout_confusion_matrix_{tag}.txt 를 읽어서
그림만 그립니다. 재학습 필요 없음 (몇 초면 끝남).

셀 표시 형식: 개수(위) + row 기준 비율(아래, 소괄호) — sklearn ConfusionMatrixDisplay 스타일
라벨: cancer.XXX / normal.control (폴더명 스타일)

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_9_plot_confusion_matrix_comparison.py --model attention
  python 07_9_plot_confusion_matrix_comparison.py --model classwise
  python 07_9_plot_confusion_matrix_comparison.py --model cnn_mean
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mil_common import MODEL_ROOT, log


MODEL_CONFIGS = {
    "attention":  {"model_gen": "gen3_attention",           "tag": "attention", "title": "Attention-MIL (gated)"},
    "classwise":  {"model_gen": "gen3_attention_classwise", "tag": "classwise", "title": "ABMIL (class-wise attention)"},
    "cnn_mean":   {"model_gen": "gen2_cnn",                 "tag": "mean",      "title": "CNN-MIL (mean pooling)"},
    "cnn_max":    {"model_gen": "gen2_cnn",                 "tag": "max",       "title": "CNN-MIL (max pooling)"},
}

# 우리 파이프라인이 confusion_matrix_*.txt 를 저장할 때 쓰는 행/열 순서 (mil_common.py CLASS_NAMES)
INTERNAL_ORDER = ["control", "NPM1", "PML_RARA", "CBFB_MYH11", "RUNX1_RUNX1T1"]

# 화면에 보여줄 순서 (폴더명 알파벳순 스타일)
DISPLAY_ORDER  = ["CBFB_MYH11", "NPM1", "PML_RARA", "RUNX1_RUNX1T1", "control"]
DISPLAY_LABELS = [f"cancer.{c}" if c != "control" else "normal.control" for c in DISPLAY_ORDER]


def reorder_matrix(cm: np.ndarray) -> np.ndarray:
    idx = [INTERNAL_ORDER.index(c) for c in DISPLAY_ORDER]
    return cm[np.ix_(idx, idx)]


def plot_one(ax, cm: np.ndarray, title: str):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > 0.6 else "black"
            ax.text(j, i - 0.12, f"{count}", ha="center", va="center",
                    color=color, fontsize=10)
            ax.text(j, i + 0.18, f"({pct:.2f})", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DISPLAY_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description="CV/Holdout confusion matrix 비교 플롯")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--run_id", type=str, default="latest")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    save_dir = MODEL_ROOT / cfg["model_gen"] / "soeun" / args.run_id / "artifacts"
    tag = cfg["tag"]

    cv_path = save_dir / f"confusion_matrix_{tag}.txt"
    holdout_path = save_dir / f"holdout_confusion_matrix_{tag}.txt"

    log(f"CV 파일       : {cv_path}")
    log(f"Holdout 파일  : {holdout_path}")

    if not cv_path.exists() or not holdout_path.exists():
        log("[ERROR] 파일이 없습니다. --model / --run_id 를 확인하세요.")
        return

    cv_cm = reorder_matrix(np.loadtxt(cv_path, dtype=int))
    holdout_cm = reorder_matrix(np.loadtxt(holdout_path, dtype=int))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_one(axes[0], cv_cm, "Confusion Matrix (5-fold CV)")
    plot_one(axes[1], holdout_cm, "Confusion Matrix (Holdout)")
    fig.suptitle(cfg["title"])
    fig.tight_layout()

    out_path = save_dir / f"confusion_matrix_comparison_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"저장 완료 → {out_path}")


if __name__ == "__main__":
    main()
