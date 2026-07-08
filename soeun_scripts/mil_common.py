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


def compute_class_weights(labels, n_classes: int = N_CLASSES):
    """CrossEntropyLoss 용 class weight (inverse frequency, 평균 1로 정규화)."""
    import torch
    counts = np.array([max((labels == c).sum(), 1) for c in range(n_classes)], dtype=np.float64)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
