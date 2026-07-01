#!/usr/bin/env bash
# Creates 20% holdout data, then applies 5-fold stratified cross-validation splits (80:20) on remaining 80%.
# Stratification is by label (normal vs cancer) to handle class imbalance.
# organized_data must be flat: all patient folders directly inside it,
# named like normal.control.AEC or cancer.CBFB_MYH11.AQK.
# Outputs:
#   data/cv_splits/holdout/ with holdout_patients.txt (one folder name per line)
#   data/cv_splits/fold_1/ ... fold_5/, each with train_patients.txt and test_patients.txt
# These split files are model-agnostic — all models read the same folds.

set -euo pipefail

SEED=42
HOLDOUT_RATIO=0.155
N_FOLDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ORGANIZED="/home/sp00001/blood_mil_project/organized_data"
CV_DIR="/home/sp00001/blood_mil_project/cv_splits"
HOLDOUT_DATA_DIR="/home/sp00001/blood_mil_project/holdout_data"

if [[ ! -d "$ORGANIZED" ]]; then
    echo "ERROR: organized_data not found at $ORGANIZED"
    echo "Run 01_organize_the_data.sh first."
    exit 1
fi

echo "Reading patients from: $ORGANIZED"
echo "Writing splits to:     $CV_DIR"
echo "Seed: $SEED  |  Holdout: $HOLDOUT_RATIO  |  Folds: $N_FOLDS  |  Stratified: yes"
echo ""

# Collect patient folder names, separated by label prefix
mapfile -t NORMAL_PATIENTS < <(
    find "$ORGANIZED" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
    | grep '^normal\.' | sort
)
mapfile -t CANCER_PATIENTS < <(
    find "$ORGANIZED" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
    | grep '^cancer\.' | sort
)

echo "Normal patients: ${#NORMAL_PATIENTS[@]}"
echo "Cancer patients: ${#CANCER_PATIENTS[@]}"
echo "Total:           $(( ${#NORMAL_PATIENTS[@]} + ${#CANCER_PATIENTS[@]} ))"
echo ""

# Hand off to Python for holdout extraction + shuffle + fold assignment + file writing
python3 - "$SEED" "$HOLDOUT_RATIO" "$N_FOLDS" "$CV_DIR" "$ORGANIZED" "$HOLDOUT_DATA_DIR" \
    "${NORMAL_PATIENTS[@]}" "---SPLIT---" "${CANCER_PATIENTS[@]}" <<'PYEOF'
import sys, random, os

seed           = int(sys.argv[1])
holdout_ratio  = float(sys.argv[2])
n_folds        = int(sys.argv[3])
cv_dir         = sys.argv[4]
organized_dir  = sys.argv[5]
holdout_data_dir = sys.argv[6]
rest           = sys.argv[7:]

sep    = rest.index("---SPLIT---")
normal = rest[:sep]
cancer = rest[sep + 1:]

rng = random.Random(seed)
rng.shuffle(normal)
rng.shuffle(cancer)

def extract_holdout(patients, ratio, rng):
    """Extract holdout_ratio fraction while maintaining balance."""
    n_holdout = max(1, int(len(patients) * ratio))
    holdout = patients[:n_holdout]
    remaining = patients[n_holdout:]
    return holdout, remaining

normal_holdout, normal_cv = extract_holdout(normal, holdout_ratio, rng)
cancer_holdout, cancer_cv = extract_holdout(cancer, holdout_ratio, rng)

def split_into_equal_folds(patients, n_folds):
    """Divide patients into n_folds of equal size, removing extras if not divisible."""
    fold_size = len(patients) // n_folds
    folds = [[] for _ in range(n_folds)]

    for i, p in enumerate(patients[:fold_size * n_folds]):
        folds[i % n_folds].append(p)
    return folds

normal_folds = split_into_equal_folds(normal_cv, n_folds)
cancer_folds = split_into_equal_folds(cancer_cv, n_folds)

os.makedirs(cv_dir, exist_ok=True)
os.makedirs(holdout_data_dir, exist_ok=True)

# Write holdout split
holdout_patients = sorted(normal_holdout + cancer_holdout)
with open(os.path.join(holdout_data_dir, "holdout_patients.txt"), "w") as f:
    f.write("\n".join(holdout_patients) + "\n")

holdout_n = len(normal_holdout)
holdout_c = len(cancer_holdout)
print(f"holdout:  {len(holdout_patients)} patients (normal={holdout_n}, cancer={holdout_c})")

# Write CV folds
for fold in range(n_folds):
    fold_dir = os.path.join(cv_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    test_patients = sorted(normal_folds[fold] + cancer_folds[fold])
    train_patients = sorted(
        [p for i, grp in enumerate(normal_folds) if i != fold for p in grp] +
        [p for i, grp in enumerate(cancer_folds) if i != fold for p in grp]
    )

    with open(os.path.join(fold_dir, "train_patients.txt"), "w") as f:
        f.write("\n".join(train_patients) + "\n")
    with open(os.path.join(fold_dir, "test_patients.txt"), "w") as f:
        f.write("\n".join(test_patients) + "\n")

    test_n  = sum(1 for p in test_patients  if p.startswith("normal."))
    test_c  = sum(1 for p in test_patients  if p.startswith("cancer."))
    train_n = sum(1 for p in train_patients if p.startswith("normal."))
    train_c = sum(1 for p in train_patients if p.startswith("cancer."))

    print(f"fold_{fold+1}: "
          f"train={len(train_patients)} (normal={train_n}, cancer={train_c})  "
          f"test={len(test_patients)} (normal={test_n}, cancer={test_c})")

print(f"\nDone. Splits written to: {cv_dir}")
PYEOF
