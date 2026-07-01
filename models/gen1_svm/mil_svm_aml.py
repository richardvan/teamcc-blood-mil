"""
AML Cytomorphology - Patient-level 5-class MIL-SVM
====================================================

Approach: embedding-space MIL (Amores, 2013 taxonomy)
  1. Extract per-cell CNN features (frozen, pretrained ResNet18)
  2. Aggregate per-patient into ONE bag descriptor via mean+max pooling
  3. Train an SVM at the patient (bag) level
     - model selection via the 5 CV fold columns already provided in metadata.csv
     - final evaluation exactly once on the untouched holdout set (is_holdout==True)

Why NOT classic mi-SVM / MI-SVM (QP-based instance-selection MIL):
  - The `misvm` package is unmaintained and frequently breaks on modern Python/cvxopt combos.
  - Its core assumption -- "a bag is positive iff at least one instance is positive" -- does not
    match this dataset. A patient's genetic subtype is a near-uniform property of most of their
    cells, so bags here are closer to homogeneous than the heterogeneous bags mi-SVM was designed
    for. Embedding-space pooling is the standard, more defensible substitute for this structure.

BEFORE RUNNING: fix IMAGE_ROOT / IMAGE_EXTS in the CONFIG block below to match where your
cell images actually live on disk (organized_data/ vs raw_data/, jpg vs tiff, etc).
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------------------------------------------------------------
# 0. CONFIG — EDIT THESE TO MATCH YOUR FILESYSTEM
# ------------------------------------------------------------------
METADATA_CSV = "metadata.csv"
IMAGE_ROOT   = "organized_data"          # <-- root folder containing one subfolder per patient
IMAGE_EXTS   = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS      = 5
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Load metadata & derive the 5-class label from the folder string
#    e.g. "cancer.CBFB_MYH11.AQK" -> "CBFB_MYH11" ; "normal.control.DNX" -> "control"
# ------------------------------------------------------------------
meta = pd.read_csv(METADATA_CSV)

def parse_label(folder: str) -> str:
    parts = folder.split(".")
    return "control" if parts[0] == "normal" else parts[1]

meta["label"] = meta["folder"].apply(parse_label)
assert meta["label"].nunique() == 5, f"Expected 5 classes, got {meta['label'].unique()}"

le = LabelEncoder().fit(meta["label"])
meta["y"] = le.transform(meta["label"])
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ------------------------------------------------------------------
# 2. Cell-level image feature extractor (frozen ResNet18, ImageNet weights)
# ------------------------------------------------------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()  # -> 512-dim embedding
resnet.eval().to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def embed_patient(folder_name: str, batch_size: int = 64) -> np.ndarray:
    """Returns (n_cells, 512) embedding matrix for one patient's cell images."""
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(IMAGE_ROOT, folder_name, ext)))
    if not paths:
        raise FileNotFoundError(
            f"No images found for '{folder_name}' under {IMAGE_ROOT}. "
            f"Check IMAGE_ROOT / IMAGE_EXTS in the CONFIG block."
        )

    feats, batch = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) == batch_size:
            x = torch.stack(batch).to(DEVICE)
            feats.append(resnet(x).cpu().numpy())
            batch = []
    if batch:
        x = torch.stack(batch).to(DEVICE)
        feats.append(resnet(x).cpu().numpy())
    return np.concatenate(feats, axis=0)

# ------------------------------------------------------------------
# 3. Build bag (patient) descriptors: mean-pool concat max-pool -> 1024-dim
# ------------------------------------------------------------------
def bag_descriptor(cell_feats: np.ndarray) -> np.ndarray:
    return np.concatenate([cell_feats.mean(axis=0), cell_feats.max(axis=0)])

print("\nExtracting per-patient bag features (one pass over every cell image)...")
bag_feats = {}
for folder in meta["folder"]:
    cell_feats = embed_patient(folder)
    bag_feats[folder] = bag_descriptor(cell_feats)
    print(f"  {folder}: {cell_feats.shape[0]} cells -> bag vector {bag_feats[folder].shape}")

X_all = np.stack([bag_feats[f] for f in meta["folder"]])
y_all = meta["y"].values

# ------------------------------------------------------------------
# 4. Split: is_holdout defines the untouched test set.
#    fold_1..fold_5 _status columns define the CV scheme on the train partition.
# ------------------------------------------------------------------
holdout_mask = meta["is_holdout"].astype(bool).values
X_train, y_train = X_all[~holdout_mask], y_all[~holdout_mask]
X_hold,  y_hold  = X_all[holdout_mask],  y_all[holdout_mask]
train_meta = meta.loc[~holdout_mask].reset_index(drop=True)

print(f"\nTrain patients: {len(train_meta)} | Holdout patients: {holdout_mask.sum()}")

# Build PredefinedSplit's test_fold array from the 5 fold_status columns.
# Rows with no assignment in any fold (blank) get -1 -> always train, never held out in CV.
test_fold = np.full(len(train_meta), -1)
for k in range(1, N_FOLDS + 1):
    col = f"fold_{k}_status"
    is_test_k = (train_meta[col] == "test").values
    test_fold[is_test_k] = k - 1
ps = PredefinedSplit(test_fold)

# ------------------------------------------------------------------
# 5. Scale + grid-search SVM hyperparameters using the provided CV folds
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_hold_s  = scaler.transform(X_hold)

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf", "linear"],
}
grid = GridSearchCV(
    SVC(class_weight="balanced", random_state=RANDOM_STATE),
    param_grid,
    cv=ps,
    scoring="balanced_accuracy",
    n_jobs=-1,
)
grid.fit(X_train_s, y_train)
print("\nBest CV params:", grid.best_params_)
print("Best CV balanced accuracy:", grid.best_score_)

# ------------------------------------------------------------------
# 6. Final holdout evaluation (touched exactly once)
# ------------------------------------------------------------------
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_hold_s)

print("\n=== HOLDOUT RESULTS ===")
print(classification_report(y_hold, y_pred, target_names=le.classes_))
print("Balanced accuracy:", balanced_accuracy_score(y_hold, y_pred))
print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_hold, y_pred))
print("Class order:", list(le.classes_))
