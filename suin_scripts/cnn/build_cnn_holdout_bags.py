#!/usr/bin/env python
"""
build_cnn_holdout_bags.py
===========================
Builds `holdout_bags` for gen2_cnn evaluation: a list of Bag objects with
.patient_id / .instances / .true_label, where .instances is a list of .tif
file path strings for that patient's cells (per CNNBagWrapper's assumption
in cnn_mil_eval.py -- CNN needs raw images since its backbone is fine-tuned,
unlike gen1_svm which can use pre-pooled frozen-feature vectors).

This is gen2_cnn-specific, not shared team infrastructure -- gen1_svm's bag
representation is fundamentally different (pooled vectors vs raw image paths),
so there's nothing to share here even in principle.

Usage as a module (for cnn_mil_eval.py's --holdout_bags_module):
    python cnn_mil_eval.py \
        --checkpoint .../cnn_mil_fold1_mean.pt \
        --holdout_bags_module build_cnn_holdout_bags \
        --model_name cnn_mil_fold1_mean \
        --output_dir /home/sp00001/blood_mil_project

That import triggers the module-level code below, which builds `holdout_bags`
using the METADATA_CSV / ORGANIZED_DIR paths set via environment variables
(see bottom of file) so this stays runnable both as a script and as an import.
"""

import glob
import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from shared_functions_V2 import SUBTYPE_TO_LABEL


@dataclass
class Bag:
    patient_id: str
    instances: List[str]   # list of .tif file paths
    true_label: int


def parse_label(folder_name: str) -> str:
    parts = folder_name.split(".")
    return "control" if parts[0] == "normal" else parts[1]


def build_holdout_bags(metadata_csv: str, organized_dir: str, image_ext: str = ".tif") -> list:
    meta = pd.read_csv(metadata_csv)
    holdout_rows = meta[meta["is_holdout"].astype(str).str.lower().isin(["true", "1"])]

    bags = []
    missing_patients = []
    for _, row in holdout_rows.iterrows():
        folder = row["folder"]
        patient_id = row["patient_id"]
        label_name = parse_label(folder)
        if label_name not in SUBTYPE_TO_LABEL:
            raise ValueError(
                f"Folder '{folder}' parsed to label '{label_name}', which isn't in "
                f"shared_functions_V2.SUBTYPE_TO_LABEL: {list(SUBTYPE_TO_LABEL)}"
            )
        true_label = SUBTYPE_TO_LABEL[label_name]

        paths = sorted(glob.glob(os.path.join(organized_dir, folder, f"*{image_ext}")))
        if not paths:
            missing_patients.append(folder)
            continue

        bags.append(Bag(patient_id=patient_id, instances=paths, true_label=true_label))

    if missing_patients:
        print(f"WARNING: {len(missing_patients)} holdout patients had no "
              f"'{image_ext}' images found and were skipped: {missing_patients}")

    print(f"Built {len(bags)} holdout bags "
          f"(class counts: {pd.Series([b.true_label for b in bags]).value_counts().to_dict()})")
    return bags


# ------------------------------------------------------------------
# Module-level `holdout_bags`, built on import, so cnn_mil_eval.py's
# `--holdout_bags_module build_cnn_holdout_bags` works via `mod.holdout_bags`.
# Paths come from env vars so this file doesn't need per-user edits.
# ------------------------------------------------------------------
METADATA_CSV = os.environ.get("METADATA_CSV", "/home/sp00001/blood_mil_project/metadata_for_multiclass.csv")
ORGANIZED_DIR = os.environ.get("ORGANIZED_DIR", "/home/sp00001/blood_mil_project/organized_data")

if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(
        f"METADATA_CSV not found at {METADATA_CSV}. Set the METADATA_CSV env var "
        f"to the correct path, e.g.: export METADATA_CSV=/path/to/metadata.csv"
    )

holdout_bags = build_holdout_bags(METADATA_CSV, ORGANIZED_DIR)


if __name__ == "__main__":
    # Quick standalone sanity check
    for b in holdout_bags[:3]:
        print(b.patient_id, b.true_label, len(b.instances), "cells")
