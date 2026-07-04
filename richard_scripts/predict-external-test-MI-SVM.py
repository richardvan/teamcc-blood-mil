import argparse
import os
import sys

import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from shared_functions import (
    MISVM,
    SVMMILWrapper,
    load_external_test_bags,
    predict_labels_and_report_performance,
)

import __main__

# Backward-compatibility for joblib files that were saved before MISVM moved into
# shared_functions.py and were therefore pickled as __main__.MISVM.
__main__.MISVM = MISVM

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(PROJECT_ROOT, "models", "gen1_svm", "misvm_v1.joblib"),
    )
    parser.add_argument("--model-gen", default="gen1_svm")
    parser.add_argument("--model-name", default="misvm_v1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = joblib.load(args.model_path)
    external_test_bags = load_external_test_bags()
    bag_df, metrics_df = predict_labels_and_report_performance(
        model=SVMMILWrapper(model),
        holdout_bags=external_test_bags,
        model_gen=args.model_gen,
        model_name=f"{args.model_name}_external_test_set",
        output_dir=PROJECT_ROOT,
    )

    print("\n=== EXTERNAL TEST SET METRICS ===")
    print(metrics_df.to_string(index=False))
