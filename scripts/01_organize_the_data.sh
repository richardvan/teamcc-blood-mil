#!/usr/bin/env bash
# Organizes raw_data into organized_data (flat — no normal/ or cancer/ subfolders).
# Copies (does not move) patient folders and generates metadata.csv.

set -euo pipefail

RAW_DATA="/home/sp00001/PKG - AML-Cytomorphology_MLL_Helmholtz_v1.zip.download/PKG - AML-Cytomorphology_MLL_Helmholtz_v1/data"
OUT_DIR="/home/sp00001/organized_data"
METADATA="$OUT_DIR/metadata.csv"

CANCER_TYPES=("CBFB_MYH11" "NPM1" "PML_RARA" "RUNX1_RUNX1T1")

echo "Source:      $RAW_DATA"
echo "Destination: $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# Write CSV header
echo "folder_name,label,cancer_subtype,patient_id,image_count,source_path" > "$METADATA"

copy_patient() {
    local src_dir="$1"      # full path to patient folder
    local label="$2"        # "normal" or "cancer"
    local subtype="$3"      # e.g. "CBFB_MYH11" or "control"
    local patient_id="$4"   # e.g. "AQK"

    local folder_name="${label}.${subtype}.${patient_id}"
    local dest="$OUT_DIR/$folder_name"

    cp -r "$src_dir" "$dest"

    local image_count
    image_count=$(find "$dest" -maxdepth 1 -name "*.tif" | wc -l | tr -d ' ')

    echo "${folder_name},${label},${subtype},${patient_id},${image_count},${src_dir}" >> "$METADATA"
    echo "  Copied: $folder_name ($image_count images)"
}

# --- normal: control only ---
echo "Processing normal (control)..."
for patient_dir in "$RAW_DATA/control"/*/; do
    patient_id="$(basename "$patient_dir")"
    copy_patient "$patient_dir" "normal" "control" "$patient_id"
done

# --- cancer: all four subtypes ---
for subtype in "${CANCER_TYPES[@]}"; do
    echo "Processing cancer ($subtype)..."
    for patient_dir in "$RAW_DATA/$subtype"/*/; do
        patient_id="$(basename "$patient_dir")"
        copy_patient "$patient_dir" "cancer" "$subtype" "$patient_id"
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
