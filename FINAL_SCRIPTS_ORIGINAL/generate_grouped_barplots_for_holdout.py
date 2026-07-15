"""
Parse the multiclass holdout_classification_report*.txt files in this folder
and generate grouped bar plots comparing models on the held-out test set.

Holdout files contain a single classification report (no per-fold repeats),
so there is only one point estimate per model/class -- no mean +/- std dev
is possible here (see generate_grouped_barplots.py for the 5-fold CV version,
which does have error bars).

Usage:
    python generate_grouped_barplots_for_holdout.py
Outputs (written to grouped_barplots_holdout/ in this folder):
    barplot_f1_by_class_holdout.png
    barplot_accuracy_by_model_holdout.png
    barplot_macro_f1_by_model_holdout.png
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(FOLDER, "grouped_barplots_holdout")

# Display name for each model, keyed by a substring of the filename.
# Edit this if you rename files or add new models.
MODEL_LABELS = {
    "SVM_multiclass": "SVM",
    "lowercase_mi_SVM": "mi-SVM",
    "uppercase_MI_SVM": "MI-SVM",
    "CNN": "CNN-MIL",
    "ABMIL": "Attention-MIL",
}

# Display order for models across all plots.
MODEL_ORDER = ["SVM", "mi-SVM", "MI-SVM", "CNN-MIL", "Attention-MIL"]

CLASS_ROW_RE = re.compile(
    r"^\s*(?P<name>.+?)\s+"
    r"(?P<precision>\d\.\d+)\s+"
    r"(?P<recall>\d\.\d+)\s+"
    r"(?P<f1>\d\.\d+)\s+"
    r"(?P<support>\d+)\s*$"
)
ACCURACY_ROW_RE = re.compile(r"^\s*accuracy\s+(\d\.\d+)\s+(\d+)\s*$", re.IGNORECASE)
NON_CLASS_NAMES = {"macro avg", "weighted avg"}


def normalize_class_name(name):
    """Strip 'cancer.'/'normal.' prefixes so class names line up across files."""
    name = name.strip()
    name = re.sub(r"^(cancer|normal)\.", "", name, flags=re.IGNORECASE)
    return name


def model_label_for_file(filename):
    for key, label in MODEL_LABELS.items():
        if key in filename:
            return label
    # Fall back to a cleaned-up version of the filename
    stem = filename.replace("holdout_classification_report.", "").replace(".txt", "")
    return stem


def find_multiclass_holdout_files():
    pattern = os.path.join(FOLDER, "holdout_classification_report.*.txt")
    files = glob.glob(pattern)
    files = [f for f in files if "binary" not in os.path.basename(f).lower()]
    return sorted(files)


def parse_file(path):
    """Return (class_rows, accuracy_row) for this holdout report (single table, no folds)."""
    with open(path) as fh:
        lines = fh.readlines()

    class_rows = []
    accuracy_row = {}

    for line in lines:
        acc_match = ACCURACY_ROW_RE.match(line)
        if acc_match:
            accuracy_row["accuracy"] = float(acc_match.group(1))
            continue

        class_match = CLASS_ROW_RE.match(line)
        if class_match:
            raw_name = class_match.group("name")
            if raw_name in NON_CLASS_NAMES:
                if raw_name == "macro avg":
                    accuracy_row["macro_f1"] = float(class_match.group("f1"))
                continue
            class_rows.append(
                {
                    "class": normalize_class_name(raw_name),
                    "precision": float(class_match.group("precision")),
                    "recall": float(class_match.group("recall")),
                    "f1": float(class_match.group("f1")),
                    "support": int(class_match.group("support")),
                }
            )

    return class_rows, accuracy_row


def load_all():
    class_records = []
    accuracy_records = []

    for path in find_multiclass_holdout_files():
        filename = os.path.basename(path)
        model = model_label_for_file(filename)
        class_rows, accuracy_row = parse_file(path)

        if not class_rows:
            print(f"WARNING: no data parsed from {filename}, skipping")
            continue

        for row in class_rows:
            row["model"] = model
            class_records.append(row)
        accuracy_row["model"] = model
        accuracy_records.append(accuracy_row)

    return pd.DataFrame(class_records), pd.DataFrame(accuracy_records)


def build_class_colors(class_order):
    """Class -> color, matching j_SVM_multiclass_holdout_eval.py's PCA plot,
    which does: colors = plt.cm.tab10(np.linspace(0, 1, len(label_categories)))
    where label_categories = sorted(meta_df['status'].unique()) (sorted WITH
    the "cancer."/"normal." prefixes still attached). Stripping those
    prefixes doesn't change the sort order, so indexing tab10 the same way
    here reproduces the exact same class -> color mapping."""
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(class_order)))
    return {cls: cmap_colors[i] for i, cls in enumerate(class_order)}


def build_model_colors(model_order):
    """Model -> color, using the same tab10-by-index scheme as build_class_colors,
    so a model's color is consistent across the accuracy/macro-F1 plots."""
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(model_order)))
    return {model: cmap_colors[i] for i, model in enumerate(model_order)}


def grouped_bar(ax, df, group_col, cat_col, value_col, cat_order=None, group_order=None, group_colors=None):
    """df has one row per (group, cat) -> plot a single point estimate per bar (no error bars)."""
    if cat_order is None:
        cat_order = sorted(df[cat_col].unique())
    if group_order is None:
        group_order = sorted(df[group_col].unique())

    n_groups = len(group_order)
    n_cats = len(cat_order)
    x = np.arange(n_cats)
    width = 0.8 / n_groups

    for i, group in enumerate(group_order):
        values = []
        for cat in cat_order:
            match = df[(df[cat_col] == cat) & (df[group_col] == group)]
            values.append(match[value_col].values[0] if not match.empty else np.nan)
        offset = (i - (n_groups - 1) / 2) * width
        color = group_colors[group] if group_colors is not None else None
        ax.bar(x + offset, values, width, label=group, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, rotation=20, ha="right")
    return ax


def main():
    class_df, accuracy_df = load_all()

    if class_df.empty:
        print("No multiclass holdout data found -- nothing to plot.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    class_order = sorted(class_df["class"].unique())
    model_order = [m for m in MODEL_ORDER if m in set(class_df["model"].unique())]
    class_colors = build_class_colors(class_order)
    model_colors = build_model_colors(model_order)

    # 1) F1-score by model, grouped by class (single holdout point estimate,
    # no error bars). Bars are colored by class, matching the PCA plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_bar(
        ax, class_df, group_col="class", cat_col="model", value_col="f1",
        cat_order=model_order, group_order=class_order,
        group_colors=class_colors,
    )
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score by model (holdout set)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "barplot_f1_by_class_holdout.png"), dpi=200)
    plt.close(fig)

    # 2) Overall accuracy by model
    bar_colors = [model_colors[m] for m in model_order]
    if not accuracy_df.empty:
        acc_summary = accuracy_df.set_index("model")["accuracy"].reindex(model_order)
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(len(model_order))
        ax.bar(x, acc_summary, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Overall accuracy by model (holdout set)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_FOLDER, "barplot_accuracy_by_model_holdout.png"), dpi=200)
        plt.close(fig)

        # 3) Macro-avg F1 by model, if we captured it
        if "macro_f1" in accuracy_df.columns:
            macro_summary = accuracy_df.set_index("model")["macro_f1"].reindex(model_order)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(x, macro_summary, color=bar_colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=20, ha="right")
            ax.set_ylabel("Macro-avg F1-score")
            ax.set_ylim(0, 1.05)
            ax.set_title("Macro-avg F1-score by model (holdout set)")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_FOLDER, "barplot_macro_f1_by_model_holdout.png"), dpi=200)
            plt.close(fig)

    print("Wrote plots to", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
