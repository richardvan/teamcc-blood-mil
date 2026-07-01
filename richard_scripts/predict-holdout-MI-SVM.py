import os
import sys
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from shared_functions import SVMMILWrapper, predict_labels_and_report_performance

from importlib.machinery import SourceFileLoader

_misvm_module = SourceFileLoader(
    "uppercase_mi_svm", os.path.join(os.path.dirname(__file__), "uppercase-MI-SVM.py")
).load_module()
load_holdout_bags = _misvm_module.load_holdout_bags

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gen1_svm", "misvm_v1.joblib")

if __name__ == "__main__":
    print(f"Loading trained model from {MODEL_PATH}", flush=True)
    model = joblib.load(MODEL_PATH)

    holdout_bags = load_holdout_bags()

    svm_wrapper = SVMMILWrapper(model)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model=svm_wrapper,
        holdout_bags=holdout_bags,
        model_gen="gen1_svm",
        model_name="misvm_v1",
        output_dir=PROJECT_ROOT,
    )

    print("\n=== HOLDOUT METRICS ===")
    print(metrics_df.to_string(index=False))
