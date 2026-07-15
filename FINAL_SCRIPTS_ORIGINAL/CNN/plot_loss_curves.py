#!/usr/bin/env python
"""
plot_loss_curves.py
=======================
Train loss + val loss vs epoch, across the 5 CV folds (thin lines) plus a
mean line, from 06a_train_cnn_cv.py's cv_history_fold{N}_{pooling}.json.

Usage:
    python plot_loss_curves.py --model_dir .../gen2_cnn --pooling mean
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_histories(model_dir, pooling):
    histories = []
    for fold in range(1, 6):
        path = os.path.join(model_dir, f"cv_history_fold{fold}_{pooling}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            histories.append(json.load(f))
    return histories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    histories = load_histories(args.model_dir, args.pooling)
    if not histories:
        print("No histories found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    all_train, all_val = [], []
    for i, h in enumerate(histories):
        epochs = [e["epoch"] for e in h]
        train_loss = [e["train_loss"] for e in h]
        val_loss = [e["val_loss"] for e in h]
        ax1.plot(epochs, train_loss, color="tab:orange", alpha=0.3, lw=1, label="fold" if i == 0 else None)
        ax2.plot(epochs, val_loss, color="tab:blue", alpha=0.3, lw=1, label="fold" if i == 0 else None)
        all_train.append(train_loss)
        all_val.append(val_loss)

    min_len = min(len(t) for t in all_train)
    mean_train = np.mean([t[:min_len] for t in all_train], axis=0)
    mean_val = np.mean([v[:min_len] for v in all_val], axis=0)
    epochs_common = list(range(1, min_len + 1))
    ax1.plot(epochs_common, mean_train, color="tab:orange", lw=2.5, label="mean")
    ax2.plot(epochs_common, mean_val, color="tab:blue", lw=2.5, label="mean")

    ax1.set_title("Train loss (5 CV folds)")
    ax2.set_title("Val loss (5 CV folds)")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=9)

    fig.suptitle(f"gen2_cnn loss curves ({args.pooling} pooling)")
    fig.tight_layout()

    out_path = args.output or os.path.join(args.model_dir, f"loss_curves_{args.pooling}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
