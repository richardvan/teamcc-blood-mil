#!/usr/bin/env python3
"""
SVM-MIL: AML Subtype Classification (5-class) — holdout 기반 최종 버전

파일 위치:
  /home/sp00001/
    shared_functions_V1.py
    blood_mil_project/
      organized_data/
      holdout_data/holdout_patients.txt
      models/gen1_svm/
        svm_mil_v1.joblib                    ← 모델 저장
      predictions/
      performance/
      soeun_scripts/
        05_2_svm_mil_subtype_holdout_v2.py   ← 이 스크립트

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 05_2_svm_mil_subtype_holdout_v2.py
"""

import sys
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    confusion_matrix, classification_report,
)

# ── 경로 자동 설정 ────────────────────────────────────────────
# 스크립트 위치 : blood_mil_project/soeun_scripts/
# PROJECT_DIR  : blood_mil_project/          (1단계 위)
# HOME_DIR     : /home/sp00001/              (2단계 위, shared_functions_V1.py 위치)
SCRIPT_DIR  = Path(__file__).resolve().parent   # .../soeun_scripts
PROJECT_DIR = SCRIPT_DIR.parent                 # .../blood_mil_project
HOME_DIR    = PROJECT_DIR.parent                # /home/sp00001

sys.path.insert(0, str(HOME_DIR))
from shared_functions_V2 import (
    predict_labels_and_report_performance,
    SVMMILWrapper as SVMMILWrapper5Class,  # V2의 SVMMILWrapper 가 이미 5-class 대응
)

# ── 레이블 정의 ───────────────────────────────────────────────
SUBTYPE_TO_LABEL = {
    "control":       0,
    "NPM1":          1,
    "PML_RARA":      2,
    "CBFB_MYH11":    3,
    "RUNX1_RUNX1T1": 4,
}
LABEL_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_LABEL.items()}
CLASS_NAMES      = [LABEL_TO_SUBTYPE[i] for i in range(len(SUBTYPE_TO_LABEL))]

# ── shared_functions 가 요구하는 Bag 인터페이스 ───────────────
@dataclass
class BagObject:
    """
    predict_labels_and_report_performance() 가 요구하는 bag 객체.
      - patient_id : str   (폴더명, e.g. "cancer.CBFB_MYH11.AQK")
      - instances  : any   (SVMMILWrapper 가 받을 입력; 여기서는 60-dim bag 벡터)
      - true_label : int
    """
    patient_id: str
    instances: np.ndarray   # shape (60,) — 이미 mean-pooled bag feature
    true_label: int


# ──────────────────────────────────────────────────────────────
# 1. Instance-level feature extraction
# ──────────────────────────────────────────────────────────────

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS     = ["contrast", "dissimilarity", "homogeneity",
                  "energy", "correlation", "ASM"]
COLOR_BINS     = 16


def extract_color_histogram(img_rgb: np.ndarray) -> np.ndarray:
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(
            img_rgb[:, :, ch], bins=COLOR_BINS, range=(0, 255), density=True
        )
        feats.append(hist)
    return np.concatenate(feats)       # 48-dim


def extract_glcm_features(img_rgb: np.ndarray) -> np.ndarray:
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=256, symmetric=True, normed=True,
    )
    feats = []
    for prop in GLCM_PROPS:
        feats.append(graycoprops(glcm, prop).mean(axis=1))
    return np.concatenate(feats)       # 12-dim


def extract_instance_features(image_path: str) -> np.ndarray:
    arr = np.asarray(Image.open(image_path).convert("RGB"))
    return np.concatenate([
        extract_color_histogram(arr),
        extract_glcm_features(arr),
    ])                                 # 60-dim


# ──────────────────────────────────────────────────────────────
# 2. Bag aggregation: mean pooling
# ──────────────────────────────────────────────────────────────

def build_bag_feature(patient_dir: Path, max_cells: int = 0, seed: int = 42) -> np.ndarray:
    tif_files = sorted(patient_dir.glob("*.tif"))
    if not tif_files:
        raise ValueError(f"No .tif images in {patient_dir}")

    if max_cells and len(tif_files) > max_cells:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(tif_files), size=max_cells, replace=False)
        tif_files = [tif_files[i] for i in idx]

    feats = []
    for f in tif_files:
        try:
            feats.append(extract_instance_features(str(f)))
        except Exception as e:
            print(f"    [WARN] skipping {f.name}: {e}")

    if not feats:
        raise ValueError(f"All images failed in {patient_dir}")

    return np.stack(feats).mean(axis=0)   # mean pooling → 60-dim


# ──────────────────────────────────────────────────────────────
# 3. 레이블 파싱
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 4. 피처 추출 + 캐싱
# ──────────────────────────────────────────────────────────────

def extract_all_bag_features(
    organized_dir: Path,
    cache_path: Path,
    max_cells_per_patient: int = 0,
) -> pd.DataFrame:
    """organized_data 전체 189명의 bag feature 추출 또는 캐시 로드."""

    if cache_path.exists():
        print(f"[cache] Loading from {cache_path}")
        data     = np.load(cache_path, allow_pickle=True)
        feat_arr = data["features"]
        df = pd.DataFrame(feat_arr, columns=[f"f{i}" for i in range(feat_arr.shape[1])])
        df["folder"] = data["folders"]
        df["label"]  = data["labels"].astype(int)
        return df

    patient_dirs = sorted([
        d for d in organized_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"Extracting features for {len(patient_dirs)} patients...")

    all_feats, all_folders, all_labels = [], [], []
    t0 = time.time()

    for i, pdir in enumerate(patient_dirs):
        label    = parse_label_from_folder(pdir.name)
        bag_feat = build_bag_feature(pdir, max_cells=max_cells_per_patient)
        all_feats.append(bag_feat)
        all_folders.append(pdir.name)
        all_labels.append(label)

        if (i + 1) % 10 == 0 or (i + 1) == len(patient_dirs):
            print(f"  [{i+1}/{len(patient_dirs)}] {pdir.name}  "
                  f"({time.time()-t0:.1f}s elapsed)")

    features = np.stack(all_feats)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
             features=features,
             folders=np.array(all_folders),
             labels=np.array(all_labels))
    print(f"[cache] Saved to {cache_path}\n")

    df = pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])])
    df["folder"] = all_folders
    df["label"]  = all_labels
    return df


# ──────────────────────────────────────────────────────────────
# 5. holdout_patients.txt 로드
# ──────────────────────────────────────────────────────────────

def load_holdout_folders(holdout_dir: Path) -> set:
    holdout_file = holdout_dir / "holdout_patients.txt"
    if not holdout_file.exists():
        raise FileNotFoundError(f"Not found: {holdout_file}")
    folders = {l.strip() for l in open(holdout_file) if l.strip()}
    print(f"[holdout] {len(folders)}명 → test set")
    return folders


# ──────────────────────────────────────────────────────────────
# 6. 분포 출력 헬퍼
# ──────────────────────────────────────────────────────────────

def print_distribution(label: str, y: np.ndarray):
    print(f"\n  {label} ({len(y)}명):")
    for idx in sorted(np.unique(y)):
        print(f"    {LABEL_TO_SUBTYPE[idx]:<20}: {(y == idx).sum()}명")


# ──────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────

def log(msg):
    """SLURM 로그에 즉시 출력되도록 flush 포함."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="SVM-MIL 5-class — holdout 기반 train/test 분할"
    )
    parser.add_argument("--max_cells_per_patient", type=int, default=0,
                        help="0 = 전체 사용")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 경로: 스크립트 위치(soeun_scripts/)에서 자동 계산
    project_dir   = PROJECT_DIR
    organized_dir = project_dir / "organized_data"
    holdout_dir   = project_dir / "holdout_data"
    output_dir    = project_dir
    cache_path    = project_dir / "cache" / "bag_features_5class.npz"

    for p, name in [(organized_dir, "organized_data"),
                    (holdout_dir,   "holdout_data")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    log("=" * 55)
    log("START: SVM-MIL Subtype Classification (5-class)")
    log(f"  Project  : {project_dir}")
    log(f"  Script   : {SCRIPT_DIR}")
    log(f"  Cache    : {cache_path}")
    log(f"  Classes  : {CLASS_NAMES}")
    log("=" * 55)

    # ── Step 1: 피처 추출 ──────────────────────────────────────
    log("[Step 1/7] 피처 추출 시작 (캐시 있으면 즉시 로드)")
    df = extract_all_bag_features(organized_dir, cache_path, args.max_cells_per_patient)
    feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    log(f"[Step 1/7] 완료 — 전체 환자: {len(df)}명, 피처 차원: {len(feat_cols)}")

    # ── Step 2: holdout 분리 ───────────────────────────────────
    log("[Step 2/7] holdout 분리 중...")
    holdout_folders = load_holdout_folders(holdout_dir)

    train_df   = df[~df["folder"].isin(holdout_folders)].reset_index(drop=True)
    holdout_df = df[ df["folder"].isin(holdout_folders)].reset_index(drop=True)

    missing = holdout_folders - set(holdout_df["folder"])
    if missing:
        log(f"  [WARN] holdout_patients.txt에 있지만 organized_data에 없는 폴더: {missing}")

    log(f"[Step 2/7] 완료 — Train: {len(train_df)}명 / Holdout(test): {len(holdout_df)}명")
    print_distribution("Train (학습)", train_df["label"].values)
    print_distribution("Test  (holdout)", holdout_df["label"].values)

    # ── Step 3: SVM 학습 ───────────────────────────────────────
    log("[Step 3/7] SVM 학습 시작...")
    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=args.seed,
        )),
    ])
    pipeline.fit(X_train, y_train)
    log("[Step 3/7] SVM 학습 완료")

    # ── Step 4: 모델 저장 ──────────────────────────────────────
    log("[Step 4/7] 모델 저장 중...")
    model_dir  = project_dir / "models" / "gen1_svm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "svm_mil_v1.joblib"
    joblib.dump(pipeline, model_path)
    log(f"[Step 4/7] 모델 저장 완료 → {model_path}")

    # ── Step 5: holdout BagObject 목록 구성 ────────────────────
    log("[Step 5/7] holdout BagObject 구성 중...")
    holdout_bags = []
    for _, row in holdout_df.iterrows():
        holdout_bags.append(BagObject(
            patient_id = row["folder"],
            instances  = row[feat_cols].values.astype(float),
            true_label = int(row["label"]),
        ))
    log(f"[Step 5/7] 완료 — holdout bag {len(holdout_bags)}개 구성")

    # ── Step 6: shared_functions_V2 평가 ─────────────────────
    log("[Step 6/7] holdout 평가 중 (shared_functions_V2)...")
    svm_wrapper = SVMMILWrapper5Class(pipeline)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model        = svm_wrapper,
        holdout_bags = holdout_bags,
        model_gen    = "gen1_svm",
        model_name   = "svm_mil_v1",
        output_dir   = str(output_dir),
    )
    log("[Step 6/7] 평가 완료")

    # ── Step 7: 5-class 상세 결과 출력 및 저장 ────────────────
    log("[Step 7/7] 5-class 상세 결과 계산 중...")
    # bag_df 는 이미 holdout 순서대로 정렬되어 있으므로 직접 사용
    y_true = bag_df["true_label"].values
    y_pred = bag_df["pred_label"].values

    f1_per_cls  = f1_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                           average=None, zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true, y_pred)
    cm          = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    print("", flush=True)
    print("=" * 60, flush=True)
    print("Holdout 평가 결과 (5-class)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Accuracy     : {acc:.3f}", flush=True)
    print(f"  F1 (macro)   : {f1_macro:.3f}", flush=True)
    print(f"  F1 (weighted): {f1_weighted:.3f}", flush=True)
    print(f"\n  클래스별 F1:", flush=True)
    for cls, f1 in zip(CLASS_NAMES, f1_per_cls):
        print(f"    {cls:<20}: {f1:.3f}", flush=True)

    print(f"\n  Confusion matrix (행=정답, 열=예측):", flush=True)
    header = "            " + "  ".join(f"{n[:6]:>6}" for n in CLASS_NAMES)
    print(header, flush=True)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:10]:<12}" + "  ".join(f"{v:>6}" for v in row), flush=True)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print(f"\n  Classification report:\n{report}", flush=True)

    # 5-class 상세 지표 JSON 추가 저장
    perf_dir = output_dir / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    detail = {
        "model_gen":    "gen1_svm",
        "model_name":   "svm_mil_v1",
        "n_train":      int(len(train_df)),
        "n_test":       int(len(holdout_df)),
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]) for cls in CLASS_NAMES},
    }
    detail_path = perf_dir / "performance_metrics_5class.gen1_svm_svm_mil_v1.json"
    detail_path.write_text(json.dumps(detail, indent=2))

    log("[Step 7/7] 완료")
    log("=" * 55)
    log("DONE: 모든 단계 완료")
    log(f"  predictions : {output_dir}/predictions/predicted_labels.gen1_svm_svm_mil_v1.csv")
    log(f"  performance : {output_dir}/performance/performance_metrics.gen1_svm_svm_mil_v1.csv")
    log(f"  performance : {detail_path}")
    log(f"  model       : {model_path}")
    log("=" * 55)
    print("\nshared_functions metrics_df:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
