"""
Figure 3 — DUNG 2 PANEL (khong phai 4): (a) Empirical coverage, (b) Mean
interval width. Moi panel dung grouped bar chart: 3 method tren truc x,
moi method co 2 cot canh nhau phan biet theo dataset (mau khac nhau).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "tables")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METHOD_COLS = {
    "GRU + CP": "main",
    "MC Dropout": "mcdropout",
    "Deep Ensemble": "deepensemble",
}
DATASETS = ["gefcom2014", "opsd"]
DATASET_LABELS = {"gefcom2014": "GEFCom2014", "opsd": "OPSD"}
DATASET_COLORS = {"gefcom2014": "tab:blue", "opsd": "tab:orange"}
NOMINAL_TARGET = 0.90


def load_stats():
    rows = []
    for name in DATASETS:
        ms = pd.read_csv(os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv"))
        for method_label, col_prefix in METHOD_COLS.items():
            cov = ms[f"{col_prefix}_coverage"]
            width = ms[f"{col_prefix}_width"]
            rows.append({
                "dataset": name, "method": method_label,
                "coverage_mean": cov.mean(), "coverage_std": cov.std(),
                "width_mean": width.mean(), "width_std": width.std(),
            })
    return pd.DataFrame(rows)


def grouped_bar(ax, stats, value_col, std_col, ylabel, title, normalize_width_per_dataset=False):
    methods = list(METHOD_COLS.keys())
    x = np.arange(len(methods))
    bar_w = 0.35

    for i, name in enumerate(DATASETS):
        sub = stats[stats["dataset"] == name].set_index("method").loc[methods]
        values = sub[value_col].to_numpy()
        errs = sub[std_col].to_numpy()
        if normalize_width_per_dataset:
            # OPSD width co thang do lon hon GEFCom2014 rat nhieu (MW vs don vi
            # noi bo) -> chuan hoa ve % so voi GRU+CP cua chinh dataset do de
            # 2 dataset hien thi duoc tren CUNG 1 truc y co y nghia so sanh.
            base = sub.loc["GRU + CP", value_col]
            values = values / base * 100
            errs = errs / base * 100
        offset = (i - 0.5) * bar_w
        ax.bar(x + offset, values, width=bar_w, yerr=errs, capsize=4,
               color=DATASET_COLORS[name], alpha=0.85, label=DATASET_LABELS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def main():
    stats = load_stats()
    stats.to_csv(os.path.join(RESULTS_DIR, "figure3_coverage_width_stats.csv"), index=False)
    print(stats.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    grouped_bar(axes[0], stats, "coverage_mean", "coverage_std",
                "Empirical coverage", "(a) Empirical coverage")
    axes[0].axhline(NOMINAL_TARGET, color="black", linestyle="--", linewidth=1,
                     label="Nominal 0.90 target", zorder=0)
    axes[0].legend(fontsize=7.5)
    axes[0].set_ylim(0.6, 1.05)

    grouped_bar(axes[1], stats, "width_mean", "width_std",
                "Mean interval width\n(% of GRU+CP width, within dataset)",
                "(b) Mean interval width (relative)", normalize_width_per_dataset=True)
    axes[1].axhline(100, color="black", linestyle=":", linewidth=1, alpha=0.5)

    fig.suptitle("Figure 3: Interval Quality Across Five Random Seeds", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "figure3_interval_quality.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nDa luu: {out_path}")

    print('\nCAPTION GOI Y (dung nguyen van):')
    print('"Figure 3. Interval quality across five random seeds, for GEFCom2014 and OPSD '
          '(grouped bars). (a) Empirical coverage of nominal 90% prediction intervals; the '
          'dashed line marks the nominal 0.90 target. (b) Mean prediction-interval width, '
          'expressed as a percentage of the GRU+CP width within each dataset to allow direct '
          'visual comparison across datasets with different load scales, illustrating the '
          'coverage-sharpness trade-off. Error bars represent one standard deviation across '
          'five seeds."')


if __name__ == "__main__":
    main()