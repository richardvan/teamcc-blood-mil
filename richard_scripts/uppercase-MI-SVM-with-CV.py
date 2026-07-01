import os
import sys
import time
import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from shared_functions import SVMMILWrapper, predict_labels_and_report_performance

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
METADATA_CSV = os.path.join(PROJECT_ROOT, "metadata.csv")
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "organized_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "gen1_svm")
MODEL_PATH = os.path.join(MODEL_DIR, "misvm_cv_v1.joblib")
IMAGE_EXT = "*.tif"
HIST_BINS = 16
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


class MISVM:
    """Multiple Instance SVM for weakly supervised learning."""

    def __init__(self, kernel="rbf", C=1.0, max_iterations=100):
        self.kernel = kernel
        self.C = C
        self.max_iterations = max_iterations
        self.svm = None
        self.selected_instances = {}

    def fit(self, bags, labels):
        """
        Train MI-SVM on bags of instances with bag-level labels.

        Args:
            bags: List of (N_i x D) instance matrices, one per patient
            labels: List of bag labels (0=normal, 1=cancer)
        """
        self.selected_instances = {}

        for i, bag in enumerate(bags):
            if labels[i] == 1:
                self.selected_instances[i] = np.mean(bag, axis=0)

        for iteration in range(self.max_iterations):
            iter_start = time.time()
            X_train, y_train = self._build_training_set(bags, labels)

            self.svm = SVC(
                kernel=self.kernel,
                C=self.C,
                class_weight="balanced",
                decision_function_shape="ovr",
            )
            self.svm.fit(X_train, y_train)

            changed = self._update_representatives(bags, labels)
            elapsed = time.time() - iter_start
            print(
                f"[MISVM] iteration {iteration + 1}/{self.max_iterations} "
                f"done in {elapsed:.1f}s, changed={changed}",
                flush=True,
            )
            if not changed:
                break

        return self

    def _build_training_set(self, bags, labels):
        """Construct training data from bags and selected representatives."""
        X_train = []
        y_train = []

        for i, bag in enumerate(bags):
            if labels[i] == 0:
                X_train.extend(bag)
                y_train.extend([0] * len(bag))
            else:
                X_train.append(self.selected_instances[i])
                y_train.append(1)

        return np.array(X_train), np.array(y_train)

    def _update_representatives(self, bags, labels):
        """Re-select representative instance per positive bag."""
        changed = False

        for i, bag in enumerate(bags):
            if labels[i] == 1:
                scores = self.svm.decision_function(bag)
                new_selected = bag[np.argmax(scores)]

                if not np.array_equal(new_selected, self.selected_instances[i]):
                    changed = True
                    self.selected_instances[i] = new_selected

        return changed

    def decision_function(self, instances):
        """
        Per-instance decision scores for one bag, matching the
        SVMMILWrapper/predict_labels_and_report_performance contract
        (which reduces these to a bag score via max itself).

        Args:
            instances: (N x D) instance matrix for a single bag

        Returns:
            Array of per-instance decision scores
        """
        return self.svm.decision_function(instances)

    def predict_bags(self, bags):
        """
        Predict bag-level labels for a list of bags using the trained SVM.

        Args:
            bags: List of instance matrices

        Returns:
            Array of predicted labels (0 or 1)
        """
        predictions = []
        for bag in bags:
            bag_score = np.max(self.decision_function(bag))
            predictions.append(1 if bag_score > 0 else 0)

        return np.array(predictions)


class PatientBag:
    """Holds one patient's instance feature matrix and bag-level label."""

    def __init__(self, patient_id, instances, true_label):
        self.patient_id = patient_id
        self.instances = instances
        self.true_label = true_label


def featurize_image(path):
    """Per-cell feature vector: per-channel color histogram + mean/std."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    feats = []
    for channel in range(3):
        channel_vals = arr[:, :, channel]
        hist, _ = np.histogram(channel_vals, bins=HIST_BINS, range=(0.0, 1.0))
        feats.append(hist.astype(np.float32) / hist.sum())
        feats.append(np.array([channel_vals.mean(), channel_vals.std()], dtype=np.float32))

    return np.concatenate(feats)


def load_patient_bag(folder_name):
    """Load and featurize every cell image for one patient into an (N x D) matrix."""
    paths = sorted(glob.glob(os.path.join(IMAGE_ROOT, folder_name, IMAGE_EXT)))
    if not paths:
        raise FileNotFoundError(f"No images found for '{folder_name}' under {IMAGE_ROOT}")
    return np.stack([featurize_image(p) for p in paths])


def load_metadata():
    """Load metadata.csv and derive the binary bag label (0=normal, 1=cancer)."""
    meta = pd.read_csv(METADATA_CSV)
    meta["label"] = (meta["status"] != 0).astype(int)
    return meta


def load_holdout_bags(meta=None):
    """Build holdout PatientBag objects from metadata.csv."""
    if meta is None:
        meta = load_metadata()
    holdout_meta = meta[meta["is_holdout"].astype(bool)].reset_index(drop=True)

    print(f"Loading {len(holdout_meta)} holdout patients...", flush=True)
    holdout_bags = []
    for i, row in enumerate(holdout_meta.itertuples(), start=1):
        holdout_bags.append(PatientBag(row.patient_id, load_patient_bag(row.folder), row.label))
        print(f"  [{i}/{len(holdout_meta)}] loaded {row.folder}", flush=True)

    return holdout_bags


def load_bags_and_labels():
    """Build train bags/labels/metadata and holdout PatientBag objects from metadata.csv."""
    meta = load_metadata()
    train_meta = meta[~meta["is_holdout"].astype(bool)].reset_index(drop=True)

    print(f"Loading {len(train_meta)} training patients...", flush=True)
    train_bags = []
    for i, row in enumerate(train_meta.itertuples(), start=1):
        train_bags.append(load_patient_bag(row.folder))
        print(f"  [{i}/{len(train_meta)}] loaded {row.folder}", flush=True)
    train_labels = train_meta["label"].tolist()

    holdout_bags = load_holdout_bags(meta)

    return train_bags, train_labels, train_meta, holdout_bags


def bag_accuracy(model, bags, labels):
    """Fraction of bags whose predicted label matches the true label."""
    preds = model.predict_bags(bags)
    return float(np.mean(np.array(preds) == np.array(labels)))


def cross_validate(train_bags, train_labels, train_meta, param_grid, n_folds=N_FOLDS):
    """
    Grid search over param_grid using the fold_k_status columns already present
    in metadata.csv. For each candidate, fits an MISVM on each fold's train split
    and evaluates bag-level accuracy on that fold's test split, then averages
    across folds. Returns the params with the best average CV accuracy.
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

            model = MISVM(
                kernel=params["kernel"], C=params["C"], max_iterations=CV_MAX_ITERATIONS
            )
            model.fit(fold_train_bags, fold_train_labels)
            score = bag_accuracy(model, fold_test_bags, fold_test_labels)
            fold_scores.append(score)
            print(
                f"[CV] params={params} fold={k}/{n_folds} "
                f"bag_accuracy={score:.4f}",
                flush=True,
            )

        avg_score = float(np.mean(fold_scores)) if fold_scores else -1.0
        cv_results.append({**params, "avg_cv_accuracy": avg_score})
        print(f"[CV] params={params} avg_cv_accuracy={avg_score:.4f}", flush=True)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score, cv_results


if __name__ == "__main__":
    train_bags, train_labels, train_meta, holdout_bags = load_bags_and_labels()

    print(f"\nRunning {len(PARAM_GRID)}-candidate grid search over {N_FOLDS} folds...", flush=True)
    best_params, best_score, cv_results = cross_validate(
        train_bags, train_labels, train_meta, PARAM_GRID
    )
    print("\n=== CV RESULTS ===", flush=True)
    for result in cv_results:
        print(result, flush=True)
    print(f"\nBest params: {best_params} (avg CV accuracy={best_score:.4f})", flush=True)

    print("\nRetraining on full training set with best params...", flush=True)
    model = MISVM(kernel=best_params["kernel"], C=best_params["C"], max_iterations=5)
    model.fit(train_bags, train_labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}", flush=True)

    svm_wrapper = SVMMILWrapper(model)
    bag_df, metrics_df = predict_labels_and_report_performance(
        model=svm_wrapper,
        holdout_bags=holdout_bags,
        model_gen="gen1_svm",
        model_name="misvm_cv_v1",
        output_dir=PROJECT_ROOT,
    )

    print("\n=== HOLDOUT METRICS ===")
    print(metrics_df.to_string(index=False))

    print("\n=== CV RESULTS (summary) ===", flush=True)
    for result in cv_results:
        print(result, flush=True)
    print(f"\nBest params: {best_params} (avg CV accuracy={best_score:.4f})", flush=True)
