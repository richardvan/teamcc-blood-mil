"""
bag_loader.py
Builds Bag objects (patient_id, instances, true_label) from:
- metadata.csv (columns: patient_id, status, folder, instance_count, is_holdout, fold_*)
- organized_data/{folder}/*.tif  (per-patient single-cell histopathology images)

Feature representation: each cell image is converted to grayscale, resized to
(image_size x image_size), and flattened to a 1D vector normalized to [0, 1].
This is the "downscale + flatten" approach chosen for the SVM baseline.

NOTE: this throws away color and fine texture detail. If SVM performance is
poor, revisit with hand-crafted features (color/texture/shape) instead of
raw downsampled pixels.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


class Bag:
    def __init__(self, patient_id, instances, true_label):
        self.patient_id = patient_id
        self.instances = instances  # shape: (n_instances, image_size*image_size)
        self.true_label = true_label


def load_patient_instances(folder_path, image_size=32):
    """Load all .tif images in a patient's folder into a (n_instances, image_size**2) array."""
    instance_vectors = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert("L")  # grayscale
        img = img.resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        instance_vectors.append(arr.flatten())

    if not instance_vectors:
        raise ValueError(f"No .tif images found in {folder_path}")

    return np.stack(instance_vectors)


def load_bags(metadata_csv, organized_data_dir, image_size=32, verbose=True):
    """
    Reads metadata_csv and organized_data_dir, returns (train_bags, holdout_bags)
    split according to the is_holdout column. Fold columns are ignored (not used for SVM).
    """
    meta = pd.read_csv(metadata_csv)

    train_bags, holdout_bags = [], []
    for _, row in meta.iterrows():
        patient_id = row["patient_id"]
        folder_path = os.path.join(organized_data_dir, row["folder"])

        if not os.path.isdir(folder_path):
            if verbose:
                print(f"WARNING: folder not found for patient {patient_id} ({folder_path}) -- skipping")
            continue

        instances = load_patient_instances(folder_path, image_size=image_size)
        bag = Bag(patient_id=patient_id, instances=instances, true_label=int(row["status"]))

        if bool(row["is_holdout"]):
            holdout_bags.append(bag)
        else:
            train_bags.append(bag)

    if verbose:
        print(f"Loaded {len(train_bags)} train bags, {len(holdout_bags)} holdout bags "
              f"(image_size={image_size}x{image_size}, feature_dim={image_size*image_size})")

    return train_bags, holdout_bags
