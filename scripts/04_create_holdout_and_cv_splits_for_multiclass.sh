#!/usr/bin/env bash
# Creates 20% holdout data, then applies 5-fold stratified cross-validation splits (80:20) on remaining 80%.
# Stratification is by the 5 subtypes (normal.control, cancer.CBFB_MYH11, cancer.NPM1, cancer.PML_RARA,
# cancer.RUNX1_RUNX1T1) to handle class imbalance across subtypes.
# organized_data must be flat: all patient folders directly inside it,
# named like normal.control.AEC or cancer.CBFB_MYH11.AQK.
# Outputs:
#   data/cv_splits_for_multiclass/holdout/ with holdout_patients.txt (one folder name per line)
#   data/cv_splits_for_multiclass/fold_1/ ... fold_5/, each with train_patients.txt and test_patients.txt
# These split files are model-agnostic — all models read the same folds.

set -euo pipefail

SEED=42
HOLDOUT_RATIO=0.155
N_FOLDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ORGANIZED="/home/sp00001/blood_mil_project/organized_data"
CV_DIR="/home/sp00001/blood_mil_project/cv_splits_for_multiclass"
HOLDOUT_DATA_DIR="/home/sp00001/blood_mil_project/holdout_data_for_multiclass"

if [[ ! -d "$ORGANIZED" ]]; then
    echo "ERROR: organized_data not found at $ORGANIZED"
    echo "Run 01_organize_the_data.sh first."
    exit 1
fi

echo "Reading patients from: $ORGANIZED"
echo "Writing splits to:     $CV_DIR"
echo "Seed: $SEED  |  Holdout: $HOLDOUT_RATIO  |  Folds: $N_FOLDS  |  Stratified: yes (by subtype)"
echo ""

# Collect patient folder names, separated by subtype (first two dot-separated fields)
SUBTYPES=(normal.control cancer.CBFB_MYH11 cancer.NPM1 cancer.PML_RARA cancer.RUNX1_RUNX1T1)

ALL_FOLDER_NAMES=$(find "$ORGANIZED" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)

PY_ARGS=()
for subtype in "${SUBTYPES[@]}"; do
    mapfile -t PATIENTS < <(echo "$ALL_FOLDER_NAMES" | grep "^${subtype}\." || true)
    echo "$subtype patients: ${#PATIENTS[@]}"
    PY_ARGS+=("---SUBTYPE---" "$subtype" "${PATIENTS[@]}")
done

TOTAL=$(echo "$ALL_FOLDER_NAMES" | grep -c -E "^($(IFS=\|; echo "${SUBTYPES[*]}"))\." || true)
echo "Total:           $TOTAL"
echo ""

# Hand off to Python for holdout extraction + shuffle + fold assignment + file writing
python3 - "$SEED" "$HOLDOUT_RATIO" "$N_FOLDS" "$CV_DIR" "$ORGANIZED" "$HOLDOUT_DATA_DIR" \
    "${PY_ARGS[@]}" <<'PYEOF'
import sys, random, os

seed              = int(sys.argv[1])
holdout_ratio     = float(sys.argv[2])
n_folds           = int(sys.argv[3])
cv_dir            = sys.argv[4]
organized_dir     = sys.argv[5]
holdout_data_dir  = sys.argv[6]
rest              = sys.argv[7:]

# Parse subtype groups delimited by ---SUBTYPE--- markers
groups = {}
order = []
i = 0
while i < len(rest):
    assert rest[i] == "---SUBTYPE---"
    subtype = rest[i + 1]
    i += 2
    patients = []
    while i < len(rest) and rest[i] != "---SUBTYPE---":
        patients.append(rest[i])
        i += 1
    groups[subtype] = patients
    order.append(subtype)

rng = random.Random(seed)
for subtype in order:
    rng.shuffle(groups[subtype])

def major_label(subtype):
    return subtype.split(".")[0]

def allocate_counts(subtypes, sizes, target_total):
    """Largest-remainder method: distribute target_total across subtypes
    proportionally to their sizes, so the sum matches target_total exactly
    (instead of losing patients to independent per-subtype floor rounding)."""
    exact = [sizes[s] * target_total / sum(sizes.values()) for s in subtypes]
    counts = {s: int(e) for s, e in zip(subtypes, exact)}
    remainder = target_total - sum(counts.values())
    fracs = sorted(
        subtypes, key=lambda s: (exact[subtypes.index(s)] - counts[s]), reverse=True
    )
    for s in fracs[:remainder]:
        counts[s] += 1
    return counts

# Group subtypes by major label (e.g. "normal", "cancer") so the overall
# holdout ratio matches what a binary normal-vs-cancer split would produce,
# then distribute each label's holdout count across its subtypes.
labels = {}
for subtype in order:
    labels.setdefault(major_label(subtype), []).append(subtype)

holdout_groups = {}
cv_groups = {}
for label, subtypes in labels.items():
    label_size = sum(len(groups[s]) for s in subtypes)
    target_total = max(1, int(label_size * holdout_ratio))
    sizes = {s: len(groups[s]) for s in subtypes}
    counts = allocate_counts(subtypes, sizes, target_total)
    for s in subtypes:
        n_holdout = counts[s]
        holdout_groups[s] = groups[s][:n_holdout]
        cv_groups[s] = groups[s][n_holdout:]

def split_into_equal_folds(patients, n_folds):
    """Divide patients into n_folds as evenly as possible; no patients are dropped
    (folds may differ in size by at most one patient when not evenly divisible)."""
    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(patients):
        folds[i % n_folds].append(p)
    return folds

subtype_folds = {subtype: split_into_equal_folds(cv_groups[subtype], n_folds) for subtype in order}

os.makedirs(cv_dir, exist_ok=True)
os.makedirs(holdout_data_dir, exist_ok=True)

# Write holdout split
holdout_patients = sorted(p for subtype in order for p in holdout_groups[subtype])
with open(os.path.join(holdout_data_dir, "holdout_patients.txt"), "w") as f:
    f.write("\n".join(holdout_patients) + "\n")

holdout_counts = ", ".join(f"{subtype}={len(holdout_groups[subtype])}" for subtype in order)
print(f"holdout:  {len(holdout_patients)} patients ({holdout_counts})")

# Write CV folds
for fold in range(n_folds):
    fold_dir = os.path.join(cv_dir, f"fold_{fold + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    test_patients = sorted(p for subtype in order for p in subtype_folds[subtype][fold])
    train_patients = sorted(
        p
        for subtype in order
        for i, grp in enumerate(subtype_folds[subtype])
        if i != fold
        for p in grp
    )

    with open(os.path.join(fold_dir, "train_patients.txt"), "w") as f:
        f.write("\n".join(train_patients) + "\n")
    with open(os.path.join(fold_dir, "test_patients.txt"), "w") as f:
        f.write("\n".join(test_patients) + "\n")

    def counts_str(patients):
        return ", ".join(
            f"{subtype}={sum(1 for p in patients if p.startswith(subtype + '.'))}"
            for subtype in order
        )

    print(f"fold_{fold+1}: "
          f"train={len(train_patients)} ({counts_str(train_patients)})  "
          f"test={len(test_patients)} ({counts_str(test_patients)})")

print(f"\nDone. Splits written to: {cv_dir}")
PYEOF
