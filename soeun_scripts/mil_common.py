#!/usr/bin/env python3
"""
mil_common.py — SVM-MIL / CNN-MIL / Attention-MIL 공용 유틸리티

베이스라인(05_2_svm_mil_subtype_holdout_v3.py)과 동일한 파일 구조를 전제로 합니다.

  /home/sp00001/
    shared_functions_V1.py
    blood_mil_project/
      organized_data/                  ← 환자별 .tif 폴더 (예: cancer.NPM1.ALA)
      holdout_data/holdout_patients.txt
      cache/
        cnn_features/{folder}.pt       ← 00_extract_cnn_features.py 가 생성
      models/
        gen1_svm/  gen2_cnn/  gen3_attention/
      predictions/
      performance/
      soeun_scripts/
        mil_common.py                  ← 이 파일
        00_extract_cnn_features.py
        06_1_cnn_mil_subtype_holdout.py
        07_1_attention_mil_subtype_holdout.py
"""

import datetime
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# ── 경로 자동 설정 (베이스라인과 동일 규칙) ───────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent   # .../soeun_scripts
PROJECT_DIR = SCRIPT_DIR.parent                 # .../blood_mil_project
HOME_DIR    = PROJECT_DIR.parent                # /home/sp00001

sys.path.insert(0, str(HOME_DIR))

ORGANIZED_DIR = PROJECT_DIR / "organized_data"
HOLDOUT_DIR   = PROJECT_DIR / "holdout_data"
CACHE_DIR     = PROJECT_DIR / "cache"
FEAT_CACHE_DIR = CACHE_DIR / "cnn_features"
MODEL_ROOT    = PROJECT_DIR / "models"
OUTPUT_DIR    = PROJECT_DIR

# ── 레이블 정의 (베이스라인과 동일) ────────────────────────────
SUBTYPE_TO_LABEL = {
    "control":       0,
    "NPM1":          1,
    "PML_RARA":      2,
    "CBFB_MYH11":    3,
    "RUNX1_RUNX1T1": 4,
}
LABEL_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_LABEL.items()}
CLASS_NAMES      = [LABEL_TO_SUBTYPE[i] for i in range(len(SUBTYPE_TO_LABEL))]
N_CLASSES        = len(CLASS_NAMES)


def log(msg: str):
    """SLURM 로그에 즉시 출력되도록 flush 포함."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_label_from_folder(folder_name: str) -> int:
    """
    "normal.control.AEC"    → 0
    "cancer.NPM1.ALA"       → 1
    "cancer.CBFB_MYH11.AOK" → 3
    """
    parts   = folder_name.split(".", 2)
    subtype = parts[1]
    if subtype not in SUBTYPE_TO_LABEL:
        raise ValueError(f"Unknown subtype '{subtype}' in '{folder_name}'")
    return SUBTYPE_TO_LABEL[subtype]


def load_holdout_folders(holdout_dir: Path = HOLDOUT_DIR) -> set:
    holdout_file = holdout_dir / "holdout_patients.txt"
    if not holdout_file.exists():
        raise FileNotFoundError(f"Not found: {holdout_file}")
    folders = {l.strip() for l in open(holdout_file) if l.strip()}
    log(f"[holdout] {len(folders)}명 → test set")
    return folders


def list_patient_dirs(organized_dir: Path = ORGANIZED_DIR):
    return sorted([
        d for d in organized_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def print_distribution(label: str, y: np.ndarray):
    print(f"\n  {label} ({len(y)}명):")
    for idx in sorted(np.unique(y)):
        print(f"    {LABEL_TO_SUBTYPE[idx]:<20}: {(y == idx).sum()}명")


# ── shared_functions_V1 이 요구하는 Bag 인터페이스 ─────────────
@dataclass
class BagObject:
    """
    predict_labels_and_report_performance() 가 요구하는 bag 객체.
      - patient_id : str
      - instances  : any   (모델의 predict_bag() 이 받는 입력)
      - true_label : int

    CNN-MIL / Attention-MIL 에서는 instances 가
    (N_instance, FEATURE_DIM) 크기의 캐시된 CNN 임베딩 텐서입니다.
    (SVM 처럼 이미 mean-pool 된 60차원 벡터가 아니라,
     "pooling 자체를 모델이 학습"하기 때문에 인스턴스 단위 그대로 넘깁니다.)
    """
    patient_id: str
    instances: object
    true_label: int


def build_bag_objects(folders, feat_cache_dir: Path = FEAT_CACHE_DIR):
    """폴더명 리스트 → BagObject 리스트 (캐시된 임베딩 로드)."""
    import torch
    bags = []
    for folder in folders:
        cache_path = feat_cache_dir / f"{folder}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"임베딩 캐시 없음: {cache_path}\n"
                f"→ 먼저 00_extract_cnn_features.py 를 실행하세요."
            )
        emb = torch.load(cache_path, map_location="cpu")   # (N, FEATURE_DIM)
        label = parse_label_from_folder(folder)
        bags.append(BagObject(patient_id=folder, instances=emb, true_label=label))
    return bags


METADATA_PATH = PROJECT_DIR / "metadata.csv"


def load_metadata(metadata_path: Path = METADATA_PATH):
    """
    metadata.csv 로드.
    필수 컬럼: patient_id, folder, is_holdout, fold_1_status ~ fold_5_status
      - is_holdout=True  → 최종 holdout(test) 환자, 어느 fold에도 포함 안 됨 (fold 컬럼 NaN)
      - is_holdout=False → fold_k_status 컬럼값이 'train' 또는 'test'
    """
    import pandas as pd
    if not metadata_path.exists():
        raise FileNotFoundError(f"Not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    required = {"patient_id", "folder", "is_holdout"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv에 필요한 컬럼이 없습니다: {missing}")
    return df


def get_holdout_folders_from_metadata(meta_df) -> set:
    """is_holdout=True 인 환자들의 folder명 집합."""
    return set(meta_df.loc[meta_df["is_holdout"] == True, "folder"])


def get_fold_split(meta_df, fold_num: int):
    """
    non-holdout 환자 중 fold_{fold_num}_status 컬럼 기준으로 train/test 폴더명 리스트 반환.
    (5-fold 중 하나. is_holdout=True 환자는 애초에 fold 대상이 아님)
    """
    col = f"fold_{fold_num}_status"
    if col not in meta_df.columns:
        raise ValueError(f"metadata.csv에 '{col}' 컬럼이 없습니다.")
    pool = meta_df.loc[meta_df["is_holdout"] == False]
    train_folders = pool.loc[pool[col] == "train", "folder"].tolist()
    test_folders  = pool.loc[pool[col] == "test",  "folder"].tolist()
    return train_folders, test_folders


def get_instance_filenames(folder: str, organized_dir: Path = ORGANIZED_DIR):
    """
    00_extract_cnn_features.py 와 동일한 정렬 규칙(sorted glob)으로
    .tif 파일명을 반환합니다. 캐시된 임베딩의 i번째 행 == 이 리스트의 i번째 파일.
    Attention 가중치를 실제 이미지 파일에 매핑할 때 사용합니다.
    """
    patient_dir = organized_dir / folder
    return [p.name for p in sorted(patient_dir.glob("*.tif"))]


def plot_multiclass_roc_grid(fold_data, class_names, n_classes, title, out_path):
    """
    2x3 그리드 ROC curve plot — 요약 패널 1개 + 클래스별 패널 5개 (5-class라 정확히 6칸).
    팀원 SVM 코드(roc_curve_plot.png)와 동일한 형식으로, Attention/CNN 어느 모델에
    써도 같은 모양의 그림이 나오도록 이 함수 하나로 통일했습니다.

    fold_data: [(y_true, y_proba), ...] 리스트.
      - 원소가 2개 이상(CV): 클래스별 패널에 fold마다 옅은 곡선 + 굵은 평균곡선
        (레이블에 AUC 평균 ± 표준편차 표시)
      - 원소가 1개(holdout): 옅은 곡선/표준편차 없이 단일 곡선 + AUC 값만 표시
    n_classes: 클래스 개수 (5)
    title: figure 제목
    out_path: 저장 경로 (.png)

    반환값: {"mean_auc_per_class": {...}, "std_auc_per_class": {...}}
      (JSON으로 같이 저장해두면 그림과 숫자가 항상 일치함)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    is_cv = len(fold_data) > 1
    mean_fpr = np.linspace(0, 1, 100)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    summary_ax = axes.flat[0]
    class_axes = axes.flat[1:1 + n_classes]

    class_mean_tpr, class_mean_auc, class_std_auc = {}, {}, {}

    for class_idx, class_name in enumerate(class_names):
        ax = class_axes[class_idx]
        tprs, aucs = [], []
        for y_true, y_proba in fold_data:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            col_sum = y_bin[:, class_idx].sum()
            if col_sum == 0 or col_sum == len(y_bin):
                continue  # 이 fold/split엔 해당 클래스가 아예 없거나 전부라 곡선 계산 불가
            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
            if is_cv:
                ax.plot(fpr, tpr, color=colors[class_idx], alpha=0.25, linewidth=1)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        if not tprs:
            class_mean_tpr[class_name] = None
            class_mean_auc[class_name] = None
            class_std_auc[class_name] = None
            ax.set_title(f"{class_name} (insufficient samples)")
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc_val = float(auc(mean_fpr, mean_tpr))
        std_auc_val = float(np.std(aucs)) if is_cv else 0.0
        class_mean_tpr[class_name] = mean_tpr
        class_mean_auc[class_name] = mean_auc_val
        class_std_auc[class_name] = std_auc_val

        label = (f"mean (AUC={mean_auc_val:.3f} ± {std_auc_val:.3f})" if is_cv
                 else f"AUC={mean_auc_val:.3f}")
        ax.plot(mean_fpr, mean_tpr, color=colors[class_idx], linewidth=2, label=label)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(class_name)
        ax.legend(fontsize=8)

    for class_idx, class_name in enumerate(class_names):
        if class_mean_tpr[class_name] is None:
            continue
        label = (f"{class_name} (AUC={class_mean_auc[class_name]:.3f} ± {class_std_auc[class_name]:.3f})"
                 if is_cv else f"{class_name} (AUC={class_mean_auc[class_name]:.3f})")
        summary_ax.plot(mean_fpr, class_mean_tpr[class_name], color=colors[class_idx],
                         linewidth=2, label=label)
    summary_ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    summary_ax.set_xlabel("False Positive Rate")
    summary_ax.set_ylabel("True Positive Rate")
    summary_ax.set_title("All subtypes (mean ROC)" if is_cv else "All subtypes")
    summary_ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_auc_per_class": class_mean_auc,
        "std_auc_per_class": class_std_auc,
    }


def plot_loss_curves(fold_train_losses, fold_best_epochs, final_train_loss, title, out_path):
    """
    학습 loss curve 플로팅.
      - fold_train_losses : [[epoch별 train_loss], ...] — CV fold마다 하나씩 (옅은 선)
      - fold_best_epochs  : fold_train_losses와 같은 길이의 리스트, 각 fold의 best_epoch
                            (early stopping으로 실제 선택된 지점을 점으로 표시)
      - final_train_loss  : 최종 모델(val 없이 고정 epoch 학습)의 epoch별 train_loss (굵은 선)
      - title, out_path   : 그래프 제목 / 저장 경로(.png)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fold_train_losses), 1)))
    for i, (losses, best_epoch) in enumerate(zip(fold_train_losses, fold_best_epochs)):
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color=colors[i], alpha=0.4, linewidth=1,
                label=f"Fold {i+1} (best_epoch={best_epoch})")
        if 1 <= best_epoch <= len(losses):
            ax.scatter([best_epoch], [losses[best_epoch - 1]], color=colors[i], s=40, zorder=5)

    if final_train_loss:
        epochs = np.arange(1, len(final_train_loss) + 1)
        ax.plot(epochs, final_train_loss, color="black", linewidth=2.5, label="Final model (no val)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss (CrossEntropy)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves_v2(fold_train_losses, fold_val_losses, title, out_path):
    """
    팀원 CNN 스타일 2패널 loss curve:
      왼쪽 = Train loss (5 CV folds), 오른쪽 = Val loss (5 CV folds)
      각 fold는 옅은 선("fold"), 굵은 선은 "mean" (모든 fold가 아직 값을 갖고 있는
      공통 epoch 구간까지만 — 제일 먼저 끝난 fold 길이에서 mean 선이 멈춤).

    fold_train_losses / fold_val_losses : [[epoch별 loss], ...] — fold 개수만큼.
    title    : figure 전체 제목 (예: "gen3_attention loss curves (gated attention)")
    out_path : 저장 경로(.png)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def _panel(ax, fold_losses, fold_color, mean_color, panel_title):
        fold_losses = [l for l in fold_losses if len(l) > 0]
        if not fold_losses:
            ax.set_title(f"{panel_title} (no data)")
            return
        for i, losses in enumerate(fold_losses):
            epochs = np.arange(1, len(losses) + 1)
            ax.plot(epochs, losses, color=fold_color, alpha=0.5, linewidth=1,
                    label="fold" if i == 0 else None)
        common_len = min(len(l) for l in fold_losses)
        mean_curve = np.mean([l[:common_len] for l in fold_losses], axis=0)
        ax.plot(np.arange(1, common_len + 1), mean_curve, color=mean_color,
                linewidth=2.5, label="mean")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(panel_title)
        ax.legend()

    _panel(axes[0], fold_train_losses, "moccasin", "darkorange", "Train loss (5 CV folds)")
    _panel(axes[1], fold_val_losses, "lightblue", "steelblue", "Val loss (5 CV folds)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def new_run_id() -> str:
    """새 실행을 구분할 타임스탬프 run_id 생성. 예: run_20260712_153045"""
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def update_latest_symlink(base_dir: Path, run_id: str):
    """
    base_dir/latest 심볼릭 링크가 base_dir/run_id 를 가리키도록 갱신.
    평가/rank-1/이미지 복사 스크립트들이 --run_id latest(기본값)로 실행되면
    항상 가장 최근에 학습된 모델을 자동으로 찾아가게 됩니다.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    latest_link = base_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_id, target_is_directory=True)
    log(f"  latest → {run_id} 로 갱신 ({base_dir / 'latest'})")


def compute_class_weights(labels, n_classes: int = N_CLASSES):
    """CrossEntropyLoss 용 class weight (inverse frequency, 평균 1로 정규화)."""
    import torch
    counts = np.array([max((labels == c).sum(), 1) for c in range(n_classes)], dtype=np.float64)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
