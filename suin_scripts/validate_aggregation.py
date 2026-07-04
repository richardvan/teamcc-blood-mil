"""
validate_aggregation.py
Compares 'max' vs 'frac_positive' aggregation WITHOUT touching holdout_bags.

Splits train_bags (161 patients) into a sub-training set and a validation set,
fits the SVM only on the sub-training set, and evaluates both aggregation
strategies on the validation set. Holdout stays untouched until a final
aggregation+threshold choice is locked in.
"""

import os
import sys
import numpy as np

sys.path.insert(0, "/home/sp00001/blood_mil_project/suin_script")
from bag_loader import load_bags
from shared_functions_V1 import SVMMILWrapper, predict_labels_and_report_performance
from run_gen1_svm import fit_svm

PROJECT_ROOT = "/home/sp00001/blood_mil_project"
RANDOM_SEED = 42

train_bags, holdout_bags = load_bags(
    metadata_csv=os.path.join(PROJECT_ROOT, "metadata.csv"),
    organized_data_dir=os.path.join(PROJECT_ROOT, "organized_data"),
    image_size=32,
)

# --- patient-level, stratified train/val split (not touching holdout_bags) ---
rng = np.random.default_rng(RANDOM_SEED)
pos_bags = [b for b in train_bags if b.true_label == 1]
neg_bags = [b for b in train_bags if b.true_label == 0]
rng.shuffle(pos_bags)
rng.shuffle(neg_bags)

def split(bags, val_frac=0.2):
    n_val = max(1, int(len(bags) * val_frac))
    return bags[n_val:], bags[:n_val]

pos_sub, pos_val = split(pos_bags)
neg_sub, neg_val = split(neg_bags)
sub_train_bags = pos_sub + neg_sub
val_bags = pos_val + neg_val

print(f"sub_train: {len(sub_train_bags)} ({sum(b.true_label for b in sub_train_bags)} positive)")
print(f"val:       {len(val_bags)} ({sum(b.true_label for b in val_bags)} positive)")

# --- fit SVM on sub_train only ---
fitted_svm = fit_svm(sub_train_bags)

# --- evaluate both aggregations on val_bags ---
for agg in ("max", "frac_positive"):
    wrapper = SVMMILWrapper(fitted_svm, aggregation=agg)
    print(f"\n=== aggregation={agg}, threshold={wrapper.threshold} ===")
    for bag in val_bags:
        result = wrapper.predict_bag(bag.instances)
        print(f"{bag.patient_id}: true={bag.true_label} pred={result['pred_label']} "
              f"score={result['pred_score']:.4f}")

    bag_df, metrics_df = predict_labels_and_report_performance(
        model=wrapper,
        holdout_bags=val_bags,  # reusing this function on val_bags, NOT the real holdout
        model_gen="gen1_svm_valcheck",
        model_name=f"agg_{agg}",
        output_dir=os.path.join(PROJECT_ROOT, "validation_checks"),
    )
    print(metrics_df)
