"""
shared_functions.py
Shared utilities for MIL evaluation across SVM, CNN, and Transformer generations.

Core idea:
- The evaluation function depends on a common interface, not a common model class.
- Each fitted model should expose predict_bag(bag_instances) -> dict with:
    {
      "pred_score": float,
      "pred_label": int
    }
- Wrapper/adaptor classes can be used to make different model families conform.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score


class BaseMILWrapper:
    """Abstract interface for MIL model wrappers."""
    def predict_bag(self, bag_instances):
        raise NotImplementedError("Subclasses must implement predict_bag(bag_instances)")


class SVMMILWrapper(BaseMILWrapper):
    """
    Wrapper for Gen 1 SVM-style MIL models.
    Expects a fitted model with decision_function(instances).
    """
    def __init__(self, fitted_model):
        self.model = fitted_model

    def predict_bag(self, bag_instances):
        scores = self.model.decision_function(bag_instances)
        pred_score = float(np.max(scores))
        pred_label = int(pred_score > 0)
        return {
            "pred_score": pred_score,
            "pred_label": pred_label,
        }


class CNNMILWrapper(BaseMILWrapper):
    """
    Wrapper for Gen 2 CNN-based MIL models.
    Replace the body of predict_bag() with the actual inference logic for your CNN.
    """
    def __init__(self, fitted_model, device=None):
        self.model = fitted_model
        self.device = device

    def predict_bag(self, bag_instances):
        # Example placeholder contract.
        # Typical implementation would:
        # 1. preprocess bag_instances into a tensor batch
        # 2. run self.model in eval mode
        # 3. convert logits/probabilities into pred_score and pred_label
        raise NotImplementedError("Implement CNN bag inference here")


class TransformerMILWrapper(BaseMILWrapper):
    """
    Wrapper for Gen 3 transformer-based MIL models.
    Replace the body of predict_bag() with the actual inference logic for your transformer.
    """
    def __init__(self, fitted_model, device=None):
        self.model = fitted_model
        self.device = device

    def predict_bag(self, bag_instances):
        # Example placeholder contract.
        # Typical implementation would:
        # 1. tokenize/encode or batch bag instances
        # 2. run transformer forward pass
        # 3. convert logits/attention-pooled score into pred_score and pred_label
        raise NotImplementedError("Implement transformer bag inference here")


def predict_labels_and_report_performance(model, holdout_bags, model_gen, model_name, output_dir="."):
    """
    Runs one fitted/wrapped model on holdout bags, exports standardized predicted-label CSV,
    and computes/exports performance metrics for that single model.

    Required model interface
    ------------------------
    model.predict_bag(bag_instances) -> dict with keys:
        - pred_score : float
        - pred_label : int

    Parameters
    ----------
    model : object implementing predict_bag(bag_instances)
    holdout_bags : iterable of bag objects, each with:
        - patient_id
        - instances
        - true_label
    model_gen : str
    model_name : str
    output_dir : str

    Returns
    -------
    bag_df : pandas DataFrame
    metrics_df : pandas DataFrame
    """
    tag = f"{model_gen}_{model_name}"
    pred_dir = os.path.join(output_dir, "predictions")
    perf_dir = os.path.join(output_dir, "performance")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    bag_rows = []
    for bag in holdout_bags:
        result = model.predict_bag(bag.instances)
        bag_rows.append({
            "patient_id": bag.patient_id,
            "true_label": bag.true_label,
            "pred_label": int(result["pred_label"]),
            "pred_score": float(result["pred_score"]),
            "model_gen": model_gen,
            "model_name": model_name,
        })

    bag_df = pd.DataFrame(bag_rows)
    pred_path = os.path.join(pred_dir, f"predicted_labels.{tag}.csv")
    bag_df.to_csv(pred_path, index=False)

    y_true = bag_df["true_label"]
    y_pred = bag_df["pred_label"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics_df = pd.DataFrame([{
        "model_gen": model_gen,
        "model_name": model_name,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "weighted_f1": round(weighted_f1, 4),
    }])
    metrics_path = os.path.join(perf_dir, f"performance_metrics.{tag}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    return bag_df, metrics_df


# Example usage pattern:
# svm_wrapper = SVMMILWrapper(fitted_svm)
# bag_df, metrics_df = predict_labels_and_report_performance(
#     model=svm_wrapper,
#     holdout_bags=holdout_patients,
#     model_gen="gen1_svm",
#     model_name="misvm_v1",
#     output_dir="."
# )