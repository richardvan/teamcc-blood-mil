"""
shared_functions_V2.py
Shared utilities for MIL evaluation — 5-class (multiclass) version.

V1 → V2 변경점:
  - confusion_matrix: binary (labels=[0,1]) → 5-class (labels=[0,1,2,3,4])
  - precision_score : average='binary'      → average='macro'
  - f1_score        : average='weighted'    → average='weighted' (유지)
  - metrics_df 컬럼: TP/TN/FP/FN 제거,
                     per_class_f1 (dict) 추가
  - CLASS_NAMES, LABEL_TO_SUBTYPE 공유 상수 추가

Core idea (V1 동일):
- Each fitted model must expose predict_bag(bag_instances) -> dict:
    {
      "pred_score": float,
      "pred_label": int   (0~4)
    }
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, f1_score,
    classification_report,
)

# ── 공유 레이블 상수 ──────────────────────────────────────────
SUBTYPE_TO_LABEL = {
    "control":       0,
    "NPM1":          1,
    "PML_RARA":      2,
    "CBFB_MYH11":    3,
    "RUNX1_RUNX1T1": 4,
}
LABEL_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_LABEL.items()}
CLASS_NAMES      = [LABEL_TO_SUBTYPE[i] for i in range(len(SUBTYPE_TO_LABEL))]
ALL_LABELS       = list(range(len(CLASS_NAMES)))


# ── Wrapper 인터페이스 (V1 동일) ──────────────────────────────

class BaseMILWrapper:
    """Abstract interface for MIL model wrappers."""
    def predict_bag(self, bag_instances):
        raise NotImplementedError("Subclasses must implement predict_bag(bag_instances)")


class SVMMILWrapper(BaseMILWrapper):
    """
    Wrapper for Gen 1 SVM-style MIL models (5-class).

    V1 과 달리 pred_label = argmax over decision scores (0~4).
    decision_function 은 shape (1, n_classes) 를 반환합니다 (OvR SVM).
    """
    def __init__(self, fitted_model):
        self.model = fitted_model

    def predict_bag(self, bag_instances):
        x      = np.array(bag_instances).reshape(1, -1)
        scores = self.model.decision_function(x)   # (1, 5)
        pred_label = int(np.argmax(scores))
        pred_score = float(np.max(scores))
        return {
            "pred_score": pred_score,
            "pred_label": pred_label,
        }


class CNNMILWrapper(BaseMILWrapper):
    """Wrapper for Gen 2 CNN-based MIL models (5-class placeholder)."""
    def __init__(self, fitted_model, device=None):
        self.model  = fitted_model
        self.device = device

    def predict_bag(self, bag_instances):
        raise NotImplementedError("Implement CNN bag inference here")


class TransformerMILWrapper(BaseMILWrapper):
    """Wrapper for Gen 3 transformer-based MIL models (5-class placeholder)."""
    def __init__(self, fitted_model, device=None):
        self.model  = fitted_model
        self.device = device

    def predict_bag(self, bag_instances):
        raise NotImplementedError("Implement transformer bag inference here")


# ── 핵심 평가 함수 ────────────────────────────────────────────

def predict_labels_and_report_performance(
    model,
    holdout_bags,
    model_gen,
    model_name,
    output_dir=".",
):
    """
    Runs one fitted/wrapped model on holdout bags, exports standardized
    predicted-label CSV, and computes/exports performance metrics.

    V1 과의 차이:
      - confusion_matrix: 5×5 (multiclass)
      - precision / f1  : average='macro'
      - metrics_df      : TP/TN/FP/FN 대신 per_class_f1_* 컬럼 추가
      - report.txt      : classification_report 추가 저장

    Required model interface
    ------------------------
    model.predict_bag(bag_instances) -> dict:
        - pred_score : float
        - pred_label : int  (0~4)

    Parameters
    ----------
    model        : object implementing predict_bag()
    holdout_bags : iterable of bag objects with .patient_id / .instances / .true_label
    model_gen    : str  e.g. "gen1_svm"
    model_name   : str  e.g. "svm_mil_v1"
    output_dir   : str  base folder (predictions/ and performance/ created here)

    Returns
    -------
    bag_df     : pd.DataFrame  — one row per holdout patient
    metrics_df : pd.DataFrame  — one row summary for this model
    """
    tag      = f"{model_gen}_{model_name}"
    pred_dir = os.path.join(output_dir, "predictions")
    perf_dir = os.path.join(output_dir, "performance")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    # ── 예측 ──────────────────────────────────────────────────
    bag_rows = []
    for bag in holdout_bags:
        result = model.predict_bag(bag.instances)
        bag_rows.append({
            "patient_id":      bag.patient_id,
            "true_label":      bag.true_label,
            "true_subtype":    LABEL_TO_SUBTYPE.get(bag.true_label, str(bag.true_label)),
            "pred_label":      int(result["pred_label"]),
            "pred_subtype":    LABEL_TO_SUBTYPE.get(int(result["pred_label"]), str(result["pred_label"])),
            "pred_score":      float(result["pred_score"]),
            "correct":         int(bag.true_label == int(result["pred_label"])),
            "model_gen":       model_gen,
            "model_name":      model_name,
        })

    bag_df    = pd.DataFrame(bag_rows)
    pred_path = os.path.join(pred_dir, f"predicted_labels.{tag}.csv")
    bag_df.to_csv(pred_path, index=False)

    # ── 지표 계산 (5-class) ───────────────────────────────────
    y_true = bag_df["true_label"].values
    y_pred = bag_df["pred_label"].values

    acc         = accuracy_score(y_true, y_pred)
    prec_macro  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred,        average="weighted", zero_division=0)
    f1_macro    = f1_score(y_true, y_pred,        average="macro",    zero_division=0)

    # 클래스별 F1
    f1_per_cls = f1_score(
        y_true, y_pred,
        labels=ALL_LABELS,
        average=None,
        zero_division=0,
    )

    # 5×5 confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)

    # classification report 텍스트 저장
    report = classification_report(
        y_true, y_pred,
        labels=ALL_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report_path = os.path.join(perf_dir, f"classification_report.{tag}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ── metrics_df 구성 ───────────────────────────────────────
    row = {
        "model_gen":    model_gen,
        "model_name":   model_name,
        "n_holdout":    len(bag_df),
        "accuracy":     round(float(acc),         4),
        "precision_macro": round(float(prec_macro), 4),
        "f1_macro":     round(float(f1_macro),    4),
        "weighted_f1":  round(float(f1_weighted), 4),
    }
    # 클래스별 F1 컬럼 추가
    for cls_name, f1_val in zip(CLASS_NAMES, f1_per_cls):
        row[f"f1_{cls_name}"] = round(float(f1_val), 4)

    metrics_df   = pd.DataFrame([row])
    metrics_path = os.path.join(perf_dir, f"performance_metrics.{tag}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    return bag_df, metrics_df


# ── Example usage ─────────────────────────────────────────────
# svm_wrapper = SVMMILWrapper(fitted_pipeline)
# bag_df, metrics_df = predict_labels_and_report_performance(
#     model        = svm_wrapper,
#     holdout_bags = holdout_patients,   # list of BagObject
#     model_gen    = "gen1_svm",
#     model_name   = "svm_mil_v1",
#     output_dir   = "/home/sp00001/blood_mil_project",
# )
# # 출력 파일:
# #   predictions/predicted_labels.gen1_svm_svm_mil_v1.csv
# #   performance/performance_metrics.gen1_svm_svm_mil_v1.csv
# #   performance/classification_report.gen1_svm_svm_mil_v1.txt
