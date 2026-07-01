"""
run_gen1_svm.py
Fit, save, and evaluate the Gen 1 SVM MIL model, per:
- guidelines_for_MIL_model_saving_and_reporting_V1.pdf (sections 1, 3, 5, 6)
- shared_functions_V1.py (SVMMILWrapper, predict_labels_and_report_performance)

ASSUMPTION (not specified in the guideline / shared_functions.py):
Training data is built by copying each bag's label onto every instance in that bag.
This is the simplest scheme consistent with SVMMILWrapper.predict_bag(), which takes
max(decision_function(instances)) as the bag score. If your team uses a real MI-SVM
training procedure (e.g. iterative positive-instance selection), replace the
"build training data" step below accordingly — everything downstream (saving,
wrapping, evaluation) stays the same.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC

from shared_functions_V1 import SVMMILWrapper, predict_labels_and_report_performance



import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_instance_level_training_set(train_bags):
    """
    train_bags: iterable of bag objects with .instances (n_i, n_features) and .true_label (0/1)
    Returns: X (N, n_features), y (N,) with each bag's label copied onto all its instances.
    """
    X_list, y_list = [], []
    for bag in train_bags:
        X_list.append(bag.instances)
        y_list.append(np.full(len(bag.instances), bag.true_label))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def fit_svm(train_bags, **svc_kwargs):
    X, y = build_instance_level_training_set(train_bags)
    # decision_function requires probability=False (default) and kernel of choice.
    svc_kwargs.setdefault("kernel", "rbf")
    svc_kwargs.setdefault("class_weight", "balanced")  # guard against bag/instance imbalance
    model = SVC(**svc_kwargs)
    model.fit(X, y)
    return model


def save_svm(model, output_dir=".", model_name="misvm_v1"):
    # Per guideline section 1: Gen 1 SVM -> .joblib, under models/gen1_svm/
    model_dir = os.path.join(output_dir, "models", "gen1_svm")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    return model_path


def run(train_bags, holdout_bags, output_dir=".", model_name="misvm_v1"):
    # 1. fit
    fitted_svm = fit_svm(train_bags)

    # 2. save (native format, per section 1)
    model_path = save_svm(fitted_svm, output_dir=output_dir, model_name=model_name)
    print(f"Saved SVM model to: {model_path}")

    # 3. wrap + evaluate (per section 3/5)
    svm_wrapper = SVMMILWrapper(fitted_svm)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model=svm_wrapper,
        holdout_bags=holdout_bags,
        model_gen="gen1_svm",
        model_name=model_name,
        output_dir=output_dir,
    )

    print(metrics_df)
    return fitted_svm, bag_df, metrics_df


if __name__ == "__main__":
    # Plug in your actual bag-loading logic here.
    # Each bag needs: .patient_id, .instances (array-like), .true_label (0/1)
    #
    # train_bags = load_train_bags(...)
    # holdout_bags = load_holdout_bags(...)
    # run(train_bags, holdout_bags, output_dir=".", model_name="misvm_v1")
    raise NotImplementedError(
        "Load train_bags and holdout_bags, then call run(train_bags, holdout_bags)."
    )
