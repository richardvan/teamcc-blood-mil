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

import glob
import os
import time
import pandas as pd
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.svm import SVC

HIST_BINS = 16
IMAGE_ROOT = "organized_data"
IMAGE_EXT = "*.tif"
EXTERNAL_IMAGE_ROOT = "organized_data_2"
EXTERNAL_IMAGE_EXT = "*.bmp"


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


class PatientBag:
    """Holds one patient's instance feature matrix and bag-level label."""

    def __init__(self, patient_id, instances, true_label):
        self.patient_id = patient_id
        self.instances = instances
        self.true_label = true_label


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


def compute_foreground_mask(arr_float01):
    """
    Segments the cell (foreground) out of the slide/padding (background).

    arr_float01: (H, W, 3) float32 array in [0, 1].
    Returns: boolean (H, W) mask, True = foreground.

    Background occupies the large majority of the frame in both source datasets
    (bright full-slide background in one, dark padding in the other), so whichever
    side of the Otsu split is the minority of pixels is treated as foreground. This
    works regardless of whether the background is bright or dark -- do not assume
    a fixed polarity, or this silently inverts on one of the two datasets.
    """
    gray = arr_float01.mean(axis=2)
    try:
        t = threshold_otsu(gray)
    except ValueError:
        return np.ones(gray.shape, dtype=bool)  # flat/constant image -- nothing to split

    below = gray < t
    mask = below if below.mean() < 0.5 else ~below

    frac = mask.mean()
    if frac == 0.0 or frac > 0.95:
        return np.ones(gray.shape, dtype=bool)  # Otsu didn't find a sane foreground split

    return mask


def stretch_to_unit_range(channel_vals, mask):
    """
    Rescales channel_vals so that channel_vals[mask] spans [0, 1], using only the
    foreground pixels' own min/max. Normalizes away per-image/per-dataset
    brightness and exposure differences before histogramming.
    """
    fg = channel_vals[mask]
    lo, hi = fg.min(), fg.max()
    if hi - lo < 1e-6:
        return np.zeros_like(channel_vals)
    return np.clip((channel_vals - lo) / (hi - lo), 0.0, 1.0)


def featurize_image(path):
    """
    Per-cell feature vector: per-channel color histogram + mean/std, computed over
    the foreground (Otsu-masked, per-image contrast-stretched) region only. Masking
    is computed on the raw image, before stretching, since stretching is meant to
    normalize brightness within the already-identified foreground.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mask = compute_foreground_mask(arr)

    feats = []
    for channel in range(3):
        stretched = stretch_to_unit_range(arr[:, :, channel], mask)
        fg_vals = stretched[mask]
        hist, _ = np.histogram(fg_vals, bins=HIST_BINS, range=(0.0, 1.0))
        feats.append(hist.astype(np.float32) / hist.sum())
        feats.append(np.array([fg_vals.mean(), fg_vals.std()], dtype=np.float32))

    return np.concatenate(feats)


# Both dataset loaders intentionally feed into the same featurizer so .tif and .bmp
# inputs are normalized into the same feature space before training/evaluation.
def load_metadata():
    """Load metadata.csv and derive the binary bag label (0=normal, 1=cancer)."""
    meta = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "metadata.csv"))
    meta["label"] = (meta["status"] != 0).astype(int)
    return meta


def load_patient_bag(folder_name):
    """Load and featurize every cell image for one patient into an (N x D) matrix."""
    project_root = os.path.join(os.path.dirname(__file__), "..")
    paths = sorted(glob.glob(os.path.join(project_root, IMAGE_ROOT, folder_name, IMAGE_EXT)))
    if not paths:
        raise FileNotFoundError(f"No images found for '{folder_name}' under {IMAGE_ROOT}")
    return np.stack([featurize_image(p) for p in paths])


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


def load_external_metadata():
    """Load organized_data_2/metadata.csv and derive the binary bag label (0=normal, 1=cancer)."""
    project_root = os.path.join(os.path.dirname(__file__), "..")
    meta = pd.read_csv(os.path.join(project_root, EXTERNAL_IMAGE_ROOT, "metadata.csv"))
    meta["label"] = (meta["label"] != "normal").astype(int)
    return meta


def load_external_patient_bag(folder_name):
    """Load and featurize every cell image for one patient from organized_data_2."""
    project_root = os.path.join(os.path.dirname(__file__), "..")
    paths = sorted(
        glob.glob(os.path.join(project_root, EXTERNAL_IMAGE_ROOT, folder_name, EXTERNAL_IMAGE_EXT))
    )
    if not paths:
        raise FileNotFoundError(
            f"No images found for '{folder_name}' under {EXTERNAL_IMAGE_ROOT}"
        )
    return np.stack([featurize_image(p) for p in paths])


def load_external_test_bags(meta=None):
    """
    Build PatientBag objects for every patient in organized_data_2 (the C-NMC 2019
    dataset). The model was never trained or tuned on this data, so all of it
    serves as the external test set.
    """
    if meta is None:
        meta = load_external_metadata()

    print(f"Loading {len(meta)} external test patients...", flush=True)
    external_test_bags = []
    for i, row in enumerate(meta.itertuples(), start=1):
        external_test_bags.append(
            PatientBag(row.patient_id, load_external_patient_bag(row.folder_name), row.label)
        )
        print(f"  [{i}/{len(meta)}] loaded {row.folder_name}", flush=True)

    return external_test_bags


def load_bags_and_labels(meta=None):
    """Build train bags/labels/metadata from metadata.csv."""
    if meta is None:
        meta = load_metadata()
    train_meta = meta[~meta["is_holdout"].astype(bool)].reset_index(drop=True)

    print(f"Loading {len(train_meta)} training patients...", flush=True)
    train_bags = []
    for i, row in enumerate(train_meta.itertuples(), start=1):
        train_bags.append(load_patient_bag(row.folder))
        print(f"  [{i}/{len(train_meta)}] loaded {row.folder}", flush=True)
    train_labels = train_meta["label"].tolist()

    return train_bags, train_labels, train_meta


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
