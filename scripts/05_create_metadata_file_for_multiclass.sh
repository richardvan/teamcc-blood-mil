#!/usr/bin/env bash
# Generates metadata_for_multiclass.csv from organized_data/, cv_splits_for_multiclass/, and holdout_data_for_multiclass/.
# Requires 01_organize_the_data.sh, 04_create_holdout_and_cv_splits_for_multiclass.sh to have been run.
#
# Columns:
#   patient_id, status, folder, instance_count, is_holdout,
#   fold_1_status, fold_2_status, fold_3_status, fold_4_status, fold_5_status
# For holdout patients, fold_X_status columns are set to NA.
# `status` is the subtype (e.g. normal.control, cancer.CBFB_MYH11), not a binary label.

set -euo pipefail

N_FOLDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ORGANIZED="$PROJECT_DIR/organized_data"
CV_DIR="$PROJECT_DIR/cv_splits_for_multiclass"
HOLDOUT_DIR="$PROJECT_DIR/holdout_data_for_multiclass"
METADATA="$PROJECT_DIR/metadata_for_multiclass.csv"

if [[ ! -d "$ORGANIZED" ]]; then
    echo "ERROR: organized_data not found. Run 01_organize_the_data.sh first."
    exit 1
fi

if [[ ! -d "$HOLDOUT_DIR" ]]; then
    echo "ERROR: holdout_data_for_multiclass not found. Run 04_create_holdout_and_cv_splits_for_multiclass.sh first."
    exit 1
fi

for (( f=1; f<=N_FOLDS; f++ )); do
    if [[ ! -f "$CV_DIR/fold_$f/train_patients.txt" ]]; then
        echo "ERROR: cv_splits_for_multiclass/fold_$f not found. Run 04_create_holdout_and_cv_splits_for_multiclass.sh first."
        exit 1
    fi
done

echo "Building metadata from: $ORGANIZED"
echo "Using fold splits from: $CV_DIR"
echo "Using holdout from: $HOLDOUT_DIR"
echo ""

python3 - "$ORGANIZED" "$CV_DIR" "$HOLDOUT_DIR" "$METADATA" "$N_FOLDS" <<'PYEOF'
import sys, os, csv

organized = sys.argv[1]
cv_dir    = sys.argv[2]
holdout_dir = sys.argv[3]
out_path  = sys.argv[4]
n_folds   = int(sys.argv[5])

# Load holdout patients
holdout_patients = set()
holdout_file = os.path.join(holdout_dir, "holdout_patients.txt")
if os.path.exists(holdout_file):
    with open(holdout_file) as f:
        for line in f:
            name = line.strip()
            if name:
                holdout_patients.add(name)

# Load fold membership: folder_name -> {fold_num: "train"/"test"}
fold_status = {}  # fold_status[folder_name][fold] = "train" or "test"

for fold in range(1, n_folds + 1):
    for split in ("train", "test"):
        fpath = os.path.join(cv_dir, f"fold_{fold}", f"{split}_patients.txt")
        with open(fpath) as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                if name not in fold_status:
                    fold_status[name] = {}
                fold_status[name][fold] = split

# Walk organized_data and build rows
rows = []
for entry in sorted(os.listdir(organized)):
    full_path = os.path.join(organized, entry)
    if not os.path.isdir(full_path):
        continue

    parts = entry.split(".")
    if len(parts) < 3:
        continue  # skip unexpected entries

    label      = parts[0]                   # "normal" or "cancer"
    subtype    = parts[1]                   # e.g. "control", "CBFB_MYH11"
    patient_id = parts[-1]                  # last segment, e.g. "AQK"
    status     = f"{label}.{subtype}"

    instance_count = sum(
        1 for fn in os.listdir(full_path) if fn.endswith(".tif")
    )

    is_holdout = entry in holdout_patients
    folds = fold_status.get(entry, {})
    row = {
        "patient_id":    patient_id,
        "status":        status,
        "folder":        entry,
        "instance_count": instance_count,
        "is_holdout":    is_holdout,
    }
    for fold in range(1, n_folds + 1):
        if is_holdout:
            row[f"fold_{fold}_status"] = "NA"
        else:
            row[f"fold_{fold}_status"] = folds.get(fold, "")

    rows.append(row)

fieldnames = [
    "patient_id", "status", "folder", "instance_count", "is_holdout",
] + [f"fold_{f}_status" for f in range(1, n_folds + 1)]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to: {out_path}")
PYEOF
