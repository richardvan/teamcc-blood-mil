import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from shared_functions import MISVM, load_bags_and_labels

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "gen1_svm")
MODEL_PATH = os.path.join(MODEL_DIR, "misvm_cv_v1.joblib")
N_FOLDS = 5
PARAM_GRID = [
    {"kernel": "rbf", "C": 0.1},
    {"kernel": "rbf", "C": 1.0},
    {"kernel": "rbf", "C": 10.0},
    {"kernel": "linear", "C": 0.1},
    {"kernel": "linear", "C": 1.0},
    {"kernel": "linear", "C": 10.0},
]
CV_MAX_ITERATIONS = 5


def bag_auc(model, bags, labels):
    """
    ROC-AUC over bag-level scores (max instance decision score per bag) vs true labels.
    Unlike accuracy, this doesn't depend on the decision threshold being well-calibrated
    at 0, which is what let grid search pick an overconfident, poorly-generalizing C.
    """
    scores = [model.decision_function(bag).max() for bag in bags]
    return float(roc_auc_score(labels, scores))


def cross_validate(train_bags, train_labels, train_meta, param_grid, n_folds=N_FOLDS):
    """
    Grid search over param_grid using the fold_k_status columns already present
    in metadata.csv. For each candidate, fits an MISVM on each fold's train split
    and evaluates bag-level ROC-AUC on that fold's test split, then averages
    across folds. Returns the params with the best average CV AUC.
    """
    best_params = None
    best_score = -1.0
    cv_results = []

    for params in param_grid:
        fold_scores = []

        for k in range(1, n_folds + 1):
            col = f"fold_{k}_status"
            fold_train_idx = np.where((train_meta[col] == "train").values)[0]
            fold_test_idx = np.where((train_meta[col] == "test").values)[0]
            if len(fold_train_idx) == 0 or len(fold_test_idx) == 0:
                continue

            fold_train_bags = [train_bags[i] for i in fold_train_idx]
            fold_train_labels = [train_labels[i] for i in fold_train_idx]
            fold_test_bags = [train_bags[i] for i in fold_test_idx]
            fold_test_labels = [train_labels[i] for i in fold_test_idx]
            if len(set(fold_test_labels)) < 2:
                print(
                    f"[CV] params={params} fold={k}/{n_folds} skipped "
                    f"(test fold has only one class, AUC undefined)",
                    flush=True,
                )
                continue

            model = MISVM(
                kernel=params["kernel"], C=params["C"], max_iterations=CV_MAX_ITERATIONS
            )
            model.fit(fold_train_bags, fold_train_labels)
            score = bag_auc(model, fold_test_bags, fold_test_labels)
            fold_scores.append(score)
            print(
                f"[CV] params={params} fold={k}/{n_folds} "
                f"bag_auc={score:.4f}",
                flush=True,
            )

        avg_score = float(np.mean(fold_scores)) if fold_scores else -1.0
        cv_results.append({**params, "avg_cv_auc": avg_score})
        print(f"[CV] params={params} avg_cv_auc={avg_score:.4f}", flush=True)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score, cv_results


if __name__ == "__main__":
    train_bags, train_labels, train_meta = load_bags_and_labels()

    print(f"\nRunning {len(PARAM_GRID)}-candidate grid search over {N_FOLDS} folds...", flush=True)
    best_params, best_score, cv_results = cross_validate(
        train_bags, train_labels, train_meta, PARAM_GRID
    )
    print("\n=== CV RESULTS ===", flush=True)
    for result in cv_results:
        print(result, flush=True)
    print(f"\nBest params: {best_params} (avg CV AUC={best_score:.4f})", flush=True)

    print("\nRetraining on full training set with best params...", flush=True)
    model = MISVM(kernel=best_params["kernel"], C=best_params["C"], max_iterations=5)
    model.fit(train_bags, train_labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}", flush=True)
