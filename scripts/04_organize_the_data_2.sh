#!/usr/bin/env bash
# Organizes raw_data_2 (C-NMC 2019 training data) into organized_data_2
# (flat -- no normal/ or cancer/ subfolders), mirroring the layout that
# 01_organize_the_data.sh produces for the AML-Cytomorphology dataset.
#
# Source layout is per-fold, per-class flat images (not per-patient folders):
#   fold_{0,1,2}/hem/UID_H<patient>_<img>_<n>_hem.bmp  -> normal (healthy)
#   fold_{0,1,2}/all/UID_<patient>_<img>_<n>_all.bmp   -> cancer (ALL)
# This groups images by patient id (parsed from the filename) into per-patient
# folders. Patient ids were verified unique across all three folds within each
# class (hem ids never repeat across folds; all ids never repeat across
# folds), so the fold is not needed to disambiguate them.
# Kept separate from organized_data (a different dataset).

set -euo pipefail

RAW_DATA="/home/sp00001/blood_mil_project/raw_data_2/PKG - C-NMC 2019/C-NMC_training_data"
OUT_DIR="/home/sp00001/blood_mil_project/organized_data_2"
METADATA="$OUT_DIR/metadata.csv"

FOLDS=("fold_0" "fold_1" "fold_2")

echo "Source:      $RAW_DATA"
echo "Destination: $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# Write CSV header
echo "folder_name,label,cancer_subtype,patient_id,image_count,source_path" > "$METADATA"

copy_patient_group() {
    local src_dir="$1"       # fold's hem/ or all/ dir
    local label="$2"         # "normal" or "cancer"
    local subtype="$3"       # "hem" or "ALL"
    local patient_id="$4"    # e.g. "H6" (hem) or "11" (all)
    local find_pattern="$5"  # e.g. "UID_H6_*_hem.bmp" (matched case-insensitively)

    local folder_name="${label}.${subtype}.${patient_id}"
    local dest="$OUT_DIR/$folder_name"

    mkdir -p "$dest"
    find "$src_dir" -maxdepth 1 -iname "$find_pattern" -exec cp -t "$dest" {} +

    local image_count
    image_count=$(find "$dest" -maxdepth 1 -name "*.bmp" | wc -l | tr -d ' ')

    echo "${folder_name},${label},${subtype},${patient_id},${image_count},${src_dir}" >> "$METADATA"
    echo "  Copied: $folder_name ($image_count images)"
}

for fold in "${FOLDS[@]}"; do
    # --- normal: hem ---
    echo "Processing normal (hem, $fold)..."
    hem_dir="$RAW_DATA/$fold/hem"
    mapfile -t patient_ids < <(
        find "$hem_dir" -maxdepth 1 -name "*.bmp" -printf "%f\n" \
        | sed -E 's/^UID_[Hh]([0-9]+)_.*/\1/' \
        | sort -n -u
    )
    for pid in "${patient_ids[@]}"; do
        copy_patient_group "$hem_dir" "normal" "hem" "H${pid}" "UID_H${pid}_*_hem.bmp"
    done

    # --- cancer: all ---
    echo "Processing cancer (all, $fold)..."
    all_dir="$RAW_DATA/$fold/all"
    mapfile -t patient_ids < <(
        find "$all_dir" -maxdepth 1 -name "*.bmp" -printf "%f\n" \
        | sed -E 's/^UID_([0-9]+)_.*/\1/' \
        | sort -n -u
    )
    for pid in "${patient_ids[@]}"; do
        copy_patient_group "$all_dir" "cancer" "ALL" "${pid}" "UID_${pid}_*_all.bmp"
    done
done

echo ""
echo "Done. metadata.csv written to: $METADATA"
echo ""

# Summary counts
normal_count=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "normal.*" | wc -l | tr -d ' ')
cancer_count=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "cancer.*" | wc -l | tr -d ' ')
echo "Normal patients: $normal_count"
echo "Cancer patients: $cancer_count"
echo "Total:           $((normal_count + cancer_count))"
