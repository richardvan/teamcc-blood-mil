import os
import sys

import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from shared_functions import MISVM, load_bags_and_labels

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "gen1_svm")
MODEL_PATH = os.path.join(MODEL_DIR, "misvm_v1.joblib")


if __name__ == "__main__":
    train_bags, train_labels, _train_meta = load_bags_and_labels()

    model = MISVM(kernel="rbf", C=1.0, max_iterations=5)
    model.fit(train_bags, train_labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}", flush=True)
