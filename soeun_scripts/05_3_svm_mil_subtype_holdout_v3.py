#!/usr/bin/env python3
"""
SVM-MIL: AML Subtype Classification (5-class)
팀 shared_functions.py 표준 적용 버전 (v3)
=============================================

v2 → v3 변경점:
  1. 피처 추출: 자체 구현(전체픽셀 히스토그램+GLCM)
               → 팀 표준 featurize_image() (foreground-only 히스토그램+mean/std)
  2. MIL 방식: mean pooling + 일반 SVM
               → MISVM (반복적 대표 인스턴스 선택, 진짜 MI-SVM)
  3. shared_functions.py 위치: 팀 레포의 scripts/ 폴더

파일 위치:
  /home/sp00001/
    blood_mil_project/
      scripts/
        shared_functions.py   ← 팀 공유 함수 (featurize_image, MISVM 포함)
      organized_data/
      holdout_data/
        holdout_patients.txt
      models/gen1_svm/
        svm_mil_v1.joblib
      predictions/
      performance/
      soeun_scripts/
        05_3_svm_mil_subtype_holdout_v3.py  ← 이 스크립트

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 05_3_svm_mil_subtype_holdout_v3.py
  
  또는:
  python /home/sp00001/blood_mil_project/soeun_scripts/05_3_svm_mil_subtype_holdout_v3.py
"""

import sys
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    confusion_matrix, classification_report,
)

# ── 경로 자동 설정 ────────────────────────────────────────────
# 스크립트 위치 : blood_mil_project/soeun_scripts/
# PROJECT_DIR  : blood_mil_project/          (1단계 위)
# SCRIPTS_DIR  : blood_mil_project/scripts/  (팀 shared_functions 위치)
SCRIPT_DIR  = Path(__file__).resolve().parent        # .../soeun_scripts
PROJECT_DIR = SCRIPT_DIR.parent                      # .../blood_mil_project
SCRIPTS_DIR = PROJECT_DIR / "scripts"                # .../scripts

sys.path.insert(0, str(SCRIPTS_DIR))
from shared_functions import (
    featurize_image,          # 팀 표준 피처 추출
    MISVM,                    # 팀 표준 MI-SVM
    PatientBag,               # holdout bag 객체
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

# ── 논문 SCEMILA 비교 기준 ────────────────────────────────────
SCEMILA_F1 = {
    "PML_RARA":      (0.86, 0.05),
    "NPM1":          (0.75, 0.06),
    "CBFB_MYH11":    (0.69, 0.09),
    "RUNX1_RUNX1T1": (0.75, 0.15),
}


# ──────────────────────────────────────────────────────────────
# 유틸: 타임스탬프 로그 (SLURM 로그에 즉시 출력)
# ──────────────────────────────────────────────────────────────

def log(msg):
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
# 레이블 파싱
# "normal.control.AEC"    → 0
# "cancer.NPM1.ALA"       → 1
# "cancer.CBFB_MYH11.AOK" → 3
# ──────────────────────────────────────────────────────────────

def parse_label_from_folder(folder_name: str) -> int:
    parts   = folder_name.split(".", 2)
    subtype = parts[1]
    if subtype not in SUBTYPE_TO_LABEL:
        raise ValueError(f"Unknown subtype '{subtype}' in '{folder_name}'")
    return SUBTYPE_TO_LABEL[subtype]


# ──────────────────────────────────────────────────────────────
# 피처 추출 + 캐싱
#
# 팀 표준 featurize_image() 사용:
#   - Otsu 임계값으로 배경(흰색) 제거 → 전경 마스크
#   - 전경 픽셀만 기준으로 채널별 대비 정규화
#   - 채널별 히스토그램(16-bin) + mean/std
#   - 결과: 54-dim 벡터 (3채널 × (16+2))
#
# 기존 v2 (60-dim, 전체픽셀 히스토그램+GLCM) 와 다르므로
# 캐시 파일명을 별도로 분리합니다.
# ──────────────────────────────────────────────────────────────

def extract_all_bag_features(
    organized_dir: Path,
    cache_path: Path,
) -> pd.DataFrame:
    """
    organized_data 전체 환자의 bag feature 추출 또는 캐시 로드.

    각 환자 폴더의 모든 .tif 이미지에 featurize_image() 적용 →
    인스턴스 행렬 (N_cells × 54) 을 mean pooling → bag 벡터 (54-dim).

    ※ MISVM 은 mean pooling 대신 반복적 대표 인스턴스 선택을 쓰지만,
      캐시에는 mean-pooled bag 벡터를 저장해 두고 MISVM 학습 시에는
      인스턴스 행렬 전체를 별도로 로드합니다.
    """
    if cache_path.exists():
        log(f"[cache] Loading from {cache_path}")
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
    log(f"Extracting features for {len(patient_dirs)} patients "
        f"using featurize_image() (team standard)...")

    all_feats, all_folders, all_labels = [], [], []
    t0 = time.time()

    for i, pdir in enumerate(patient_dirs):
        label    = parse_label_from_folder(pdir.name)
        tif_files = sorted(pdir.glob("*.tif"))
        if not tif_files:
            log(f"  [WARN] no .tif files in {pdir.name}, skipping")
            continue

        # 각 세포 이미지 → 54-dim 벡터
        instance_feats = []
        for f in tif_files:
            try:
                instance_feats.append(featurize_image(str(f)))
            except Exception as e:
                log(f"    [WARN] skipping {f.name}: {e}")

        if not instance_feats:
            log(f"  [WARN] all images failed in {pdir.name}, skipping")
            continue

        # mean pooling → bag 벡터
        bag_feat = np.stack(instance_feats).mean(axis=0)
        all_feats.append(bag_feat)
        all_folders.append(pdir.name)
        all_labels.append(label)

        if (i + 1) % 10 == 0 or (i + 1) == len(patient_dirs):
            log(f"  [{i+1}/{len(patient_dirs)}] {pdir.name}  "
                f"({time.time()-t0:.1f}s elapsed)")

    features = np.stack(all_feats)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path,
             features=features,
             folders=np.array(all_folders),
             labels=np.array(all_labels))
    log(f"[cache] Saved to {cache_path}")

    df = pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])])
    df["folder"] = all_folders
    df["label"]  = all_labels
    return df


# ──────────────────────────────────────────────────────────────
# MISVM 학습용 인스턴스 행렬 로드
#
# MISVM 은 각 환자의 개별 세포 이미지 벡터(N_cells × 54) 가
# 필요합니다. mean-pooled 캐시와 별개로 여기서 직접 로드합니다.
# ──────────────────────────────────────────────────────────────

def load_instance_matrix(patient_dir: Path) -> np.ndarray:
    """환자 폴더의 모든 .tif → (N_cells × 54) 인스턴스 행렬 반환."""
    tif_files = sorted(patient_dir.glob("*.tif"))
    feats = []
    for f in tif_files:
        try:
            feats.append(featurize_image(str(f)))
        except Exception as e:
            log(f"    [WARN] skipping {f.name}: {e}")
    if not feats:
        raise ValueError(f"No valid images in {patient_dir}")
    return np.stack(feats)    # (N_cells, 54)


# ──────────────────────────────────────────────────────────────
# holdout_patients.txt 로드
# ──────────────────────────────────────────────────────────────

def load_holdout_folders(holdout_dir: Path) -> set:
    holdout_file = holdout_dir / "holdout_patients.txt"
    if not holdout_file.exists():
        raise FileNotFoundError(f"Not found: {holdout_file}")
    folders = {l.strip() for l in open(holdout_file) if l.strip()}
    log(f"[holdout] {len(folders)}명 → test set")
    return folders


# ──────────────────────────────────────────────────────────────
# 5-class MISVM Wrapper
#
# 팀의 MISVM 은 binary (0/1) 로 설계되어 있어요.
# 5-class 를 위해 One-vs-Rest 방식으로 5개의 MISVM 을 따로 학습합니다.
# 예측 시: 각 binary MISVM 의 decision score 중 가장 높은 클래스 선택.
# ──────────────────────────────────────────────────────────────

class MISVM5Class:
    """
    5-class MI-SVM: One-vs-Rest 방식으로 MISVM 5개를 학습.

    팀의 MISVM 은 binary 분류기이므로,
    각 클래스(0~4)를 "해당 클래스 vs 나머지" 로 5번 학습합니다.
    """

    def __init__(self, kernel="rbf", C=1.0, max_iterations=20):
        self.kernel        = kernel
        self.C             = C
        self.max_iterations = max_iterations
        self.classifiers   = {}    # {class_idx: MISVM}
        self.scaler        = StandardScaler()

    def fit(self, bags: list, labels: list):
        """
        bags  : list of (N_cells × 54) numpy arrays, 하나 per 환자
        labels: list of int (0~4)
        """
        # 스케일러는 전체 인스턴스로 fit
        all_instances = np.vstack(bags)
        self.scaler.fit(all_instances)
        scaled_bags = [self.scaler.transform(b) for b in bags]

        for cls_idx in range(len(CLASS_NAMES)):
            log(f"  MISVM fitting class {cls_idx} ({CLASS_NAMES[cls_idx]}) "
                f"vs rest...")
            binary_labels = [1 if l == cls_idx else 0 for l in labels]
            misvm = MISVM(
                kernel=self.kernel,
                C=self.C,
                max_iterations=self.max_iterations,
            )
            misvm.fit(scaled_bags, binary_labels)
            self.classifiers[cls_idx] = misvm
            log(f"  Class {cls_idx} done.")

        return self

    def decision_function(self, instances: np.ndarray) -> np.ndarray:
        """
        instances: (N_cells × 54) 인스턴스 행렬 (단일 환자)
        반환: (5,) 각 클래스의 max decision score
        """
        scaled = self.scaler.transform(instances)
        scores = []
        for cls_idx in range(len(CLASS_NAMES)):
            cls_scores = self.classifiers[cls_idx].decision_function(scaled)
            scores.append(float(np.max(cls_scores)))
        return np.array(scores)   # (5,)

    def predict_bag(self, instances: np.ndarray) -> dict:
        scores     = self.decision_function(instances)
        pred_label = int(np.argmax(scores))
        pred_score = float(np.max(scores))
        return {"pred_label": pred_label, "pred_score": pred_score}


# ──────────────────────────────────────────────────────────────
# 분포 출력 헬퍼
# ──────────────────────────────────────────────────────────────

def print_distribution(label: str, y: np.ndarray):
    print(f"\n  {label} ({len(y)}명):", flush=True)
    for idx in sorted(np.unique(y)):
        print(f"    {LABEL_TO_SUBTYPE[idx]:<20}: {(y == idx).sum()}명",
              flush=True)


# ──────────────────────────────────────────────────────────────
# 결과 출력 + 저장
# ──────────────────────────────────────────────────────────────

def report_and_save(
    y_true, y_pred, pred_scores,
    patient_ids, output_dir: Path,
):
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_cls  = f1_score(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    print(f"\n{'='*60}", flush=True)
    print("Holdout 평가 결과 (5-class MISVM)", flush=True)
    print(f"{'='*60}", flush=True)
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
        print(f"  {CLASS_NAMES[i][:10]:<12}" +
              "  ".join(f"{v:>6}" for v in row), flush=True)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print(f"\n  Classification report:\n{report}", flush=True)

    print(f"\n  논문(SCEMILA) vs MISVM 비교:", flush=True)
    print(f"  {'클래스':<20} {'SCEMILA':>12} {'MISVM':>10}", flush=True)
    print("  " + "-" * 44, flush=True)
    for cls in ["PML_RARA", "NPM1", "CBFB_MYH11", "RUNX1_RUNX1T1"]:
        s_m, s_s = SCEMILA_F1[cls]
        val      = f1_per_cls[SUBTYPE_TO_LABEL[cls]]
        print(f"  {cls:<20} {s_m:.2f}±{s_s:.2f}    {val:.3f}", flush=True)

    # ── 파일 저장 ─────────────────────────────────────────────
    pred_dir = output_dir / "predictions"
    perf_dir = output_dir / "performance"
    pred_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    # predictions CSV (팀 표준 형식)
    pd.DataFrame({
        "patient_id":   patient_ids,
        "true_label":   y_true,
        "pred_label":   y_pred,
        "pred_score":   pred_scores,
        "true_subtype": [LABEL_TO_SUBTYPE[l] for l in y_true],
        "pred_subtype": [LABEL_TO_SUBTYPE[l] for l in y_pred],
        "correct":      (np.array(y_true) == np.array(y_pred)).astype(int),
        "model_gen":    "gen1_svm",
        "model_name":   "misvm_v1",
    }).to_csv(pred_dir / "predicted_labels.gen1_svm_misvm_v1.csv", index=False)

    # performance CSV
    perf_row = {
        "model_gen":       "gen1_svm",
        "model_name":      "misvm_v1",
        "n_holdout":       len(y_true),
        "accuracy":        round(float(acc),         4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro":        round(float(f1_macro),    4),
        "weighted_f1":     round(float(f1_weighted), 4),
    }
    for cls in CLASS_NAMES:
        perf_row[f"f1_{cls}"] = round(float(f1_per_cls[SUBTYPE_TO_LABEL[cls]]), 4)

    pd.DataFrame([perf_row]).to_csv(
        perf_dir / "performance_metrics.gen1_svm_misvm_v1.csv", index=False)

    # classification report txt
    (perf_dir / "classification_report.gen1_svm_misvm_v1.txt").write_text(report)

    # 상세 JSON
    detail = {
        "model_gen":    "gen1_svm",
        "model_name":   "misvm_v1",
        "n_train":      None,   # main 에서 채워짐
        "n_holdout":    int(len(y_true)),
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "per_class_f1": {cls: float(f1_per_cls[SUBTYPE_TO_LABEL[cls]])
                         for cls in CLASS_NAMES},
    }
    return detail


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="5-class MISVM (팀 표준 featurize_image + MISVM)"
    )
    parser.add_argument("--max_iter", type=int, default=20,
                        help="MISVM 반복 횟수 (기본 20)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    project_dir   = PROJECT_DIR
    organized_dir = project_dir / "organized_data"
    holdout_dir   = project_dir / "holdout_data"
    output_dir    = project_dir
    cache_path    = project_dir / "cache" / "bag_features_v3.npz"  # v2와 별도 캐시

    for p, name in [(organized_dir, "organized_data"),
                    (holdout_dir,   "holdout_data"),
                    (SCRIPTS_DIR,   "scripts")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    log("=" * 55)
    log("START: 5-class MISVM (팀 표준 featurize_image)")
    log(f"  Project  : {project_dir}")
    log(f"  Script   : {SCRIPT_DIR}")
    log(f"  Cache    : {cache_path}")
    log(f"  Classes  : {CLASS_NAMES}")
    log(f"  max_iter : {args.max_iter}")
    log("=" * 55)

    # ── Step 1: mean-pooled 피처 추출 (캐시용) ────────────────
    log("[Step 1/6] 피처 추출 (featurize_image, foreground-aware)...")
    df = extract_all_bag_features(organized_dir, cache_path)
    log(f"[Step 1/6] 완료 — 환자: {len(df)}명, 피처: {len([c for c in df.columns if c.startswith('f') and c[1:].isdigit()])}차원")

    # ── Step 2: holdout 분리 ──────────────────────────────────
    log("[Step 2/6] holdout 분리 중...")
    holdout_folders = load_holdout_folders(holdout_dir)
    train_df   = df[~df["folder"].isin(holdout_folders)].reset_index(drop=True)
    holdout_df = df[ df["folder"].isin(holdout_folders)].reset_index(drop=True)

    log(f"[Step 2/6] 완료 — Train: {len(train_df)}명 / Holdout: {len(holdout_df)}명")
    print_distribution("Train (학습)", train_df["label"].values)
    print_distribution("Test  (holdout)", holdout_df["label"].values)

    # ── Step 3: 학습용 인스턴스 행렬 로드 ─────────────────────
    # MISVM 은 각 환자의 개별 세포 벡터 전체가 필요합니다
    log("[Step 3/6] 학습용 인스턴스 행렬 로드 중...")
    train_bags   = []
    train_labels = []
    t0 = time.time()
    for i, row in train_df.iterrows():
        pdir = organized_dir / row["folder"]
        mat  = load_instance_matrix(pdir)
        train_bags.append(mat)
        train_labels.append(int(row["label"]))
        if (len(train_bags)) % 20 == 0 or len(train_bags) == len(train_df):
            log(f"  [{len(train_bags)}/{len(train_df)}] loaded  "
                f"({time.time()-t0:.1f}s)")
    log(f"[Step 3/6] 완료 — 총 인스턴스: "
        f"{sum(b.shape[0] for b in train_bags):,}개")

    # ── Step 4: MISVM 학습 ────────────────────────────────────
    log("[Step 4/6] MISVM 학습 시작 (One-vs-Rest, 5 classifiers)...")
    model = MISVM5Class(kernel="rbf", C=1.0, max_iterations=args.max_iter)
    model.fit(train_bags, train_labels)
    log("[Step 4/6] MISVM 학습 완료")

    # ── Step 5: 모델 저장 ─────────────────────────────────────
    log("[Step 5/6] 모델 저장 중...")
    model_dir  = project_dir / "models" / "gen1_svm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "misvm_v1.joblib"
    joblib.dump(model, model_path)
    log(f"[Step 5/6] 완료 → {model_path}")

    # ── Step 6: holdout 평가 ──────────────────────────────────
    log("[Step 6/6] holdout 평가 중...")
    y_true, y_pred, pred_scores, patient_ids = [], [], [], []

    for i, row in holdout_df.iterrows():
        pdir   = organized_dir / row["folder"]
        mat    = load_instance_matrix(pdir)
        result = model.predict_bag(mat)
        y_true.append(int(row["label"]))
        y_pred.append(result["pred_label"])
        pred_scores.append(result["pred_score"])
        patient_ids.append(row["folder"])
        log(f"  [{len(y_pred)}/{len(holdout_df)}] {row['folder']} "
            f"→ pred={LABEL_TO_SUBTYPE[result['pred_label']]} "
            f"(true={LABEL_TO_SUBTYPE[int(row['label'])]})")

    log("[Step 6/6] 평가 완료")

    detail = report_and_save(
        y_true, y_pred, pred_scores, patient_ids, output_dir
    )
    detail["n_train"] = len(train_df)

    perf_dir = output_dir / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    (perf_dir / "performance_metrics_5class.gen1_svm_misvm_v1.json").write_text(
        json.dumps(detail, indent=2)
    )

    log("=" * 55)
    log("DONE")
    log(f"  predictions : {output_dir}/predictions/predicted_labels.gen1_svm_misvm_v1.csv")
    log(f"  performance : {output_dir}/performance/performance_metrics.gen1_svm_misvm_v1.csv")
    log(f"  model       : {model_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
