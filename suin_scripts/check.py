import os
import sys
import joblib
import numpy as np

sys.path.insert(0, "/home/sp00001/blood_mil_project/suin_scripts")
from bag_loader import load_bags

PROJECT_ROOT = "/home/sp00001/blood_mil_project"

model = joblib.load(os.path.join(PROJECT_ROOT, "models/gen1_svm/misvm_v1.joblib"))

train_bags, holdout_bags = load_bags(
    metadata_csv=os.path.join(PROJECT_ROOT, "metadata.csv"),
    organized_data_dir=os.path.join(PROJECT_ROOT, "organized_data"),
    image_size=32,
)

# 전체 holdout 인스턴스를 한데 모아서 decision_function 분포 확인
all_instances = np.vstack([bag.instances for bag in holdout_bags])
scores = model.decision_function(all_instances)

print(f"n_instances: {len(scores)}")
print(f"score min/max/mean: {scores.min():.4f} / {scores.max():.4f} / {scores.mean():.4f}")
print(f"fraction of instances with score > 0 (predicted positive): {(scores > 0).mean():.4f}")

print("\n--- per-patient breakdown ---")
for bag in holdout_bags:
    s = model.decision_function(bag.instances)
    print(f"{bag.patient_id}: true_label={bag.true_label}, "
          f"max_score={s.max():.4f}, frac_positive_instances={(s > 0).mean():.4f}")

    
    
print(sum(b.true_label for b in train_bags), len(train_bags))