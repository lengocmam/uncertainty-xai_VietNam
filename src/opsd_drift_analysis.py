"""
Drift Analysis cho OPSD — so sánh phân phối CALIBRATION WINDOW (60%-80%
dữ liệu) với TEST WINDOW (80%-100%), KHÔNG PHẢI train vs test.

Đây là bằng chứng TRỰC TIẾP cho giả thuyết "exchangeability bị vi phạm"
đã suy luận gián tiếp qua Ablation B, epistemic/aleatoric, và multi-seed
robustness ở các bước trước — nếu phân phối calibration và test khác
nhau đáng kể, đây chính là lý do conformal calibration (dựa trên giả
định 2 tập này TRAO ĐỔI ĐƯỢC - exchangeable) hoạt động kém hiệu quả.

Output: results/tables/opsd_drift_analysis.csv + figure so sánh phân phối.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # không cần hiển thị màn hình, chỉ lưu file
import matplotlib.pyplot as plt

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def compare_distributions(calib_values: np.ndarray, test_values: np.ndarray):
    """Trả về dict các chỉ số so sánh phân phối giữa 2 tập."""
    ks_stat, ks_p = stats.ks_2samp(calib_values, test_values)
    wdist = stats.wasserstein_distance(calib_values, test_values)
    return {
        "calibration_mean": calib_values.mean(),
        "test_mean": test_values.mean(),
        "calibration_std": calib_values.std(),
        "test_std": test_values.std(),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "wasserstein_distance": wdist,
    }


def normalize(x: np.ndarray):
    return (x - x.mean()) / (x.std() + 1e-8)


def run_drift_analysis(df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end, calib_end = int(n * 0.6), int(n * 0.8)
    calib_df = df.iloc[train_end:calib_end]
    test_df = df.iloc[calib_end:]

    print(f"Calibration window: {calib_df['timestamp'].min()} -> {calib_df['timestamp'].max()} "
          f"({len(calib_df)} dong)")
    print(f"Test window       : {test_df['timestamp'].min()} -> {test_df['timestamp'].max()} "
          f"({len(test_df)} dong)\n")

    results = []

    calib_load = calib_df[target_col].to_numpy(dtype="float64")
    test_load = test_df[target_col].to_numpy(dtype="float64")

    row = {"feature": "LOAD (raw)"}
    row.update(compare_distributions(calib_load, test_load))
    results.append(row)

    calib_load_norm = normalize(calib_load)
    test_load_norm = (test_load - calib_load.mean()) / (calib_load.std() + 1e-8)
    row = {"feature": "LOAD (normalized, theo thong ke calibration)"}
    row.update(compare_distributions(calib_load_norm, test_load_norm))
    results.append(row)

    for col in feature_cols:
        calib_vals = calib_df[col].to_numpy(dtype="float64")
        test_vals = test_df[col].to_numpy(dtype="float64")
        row = {"feature": col}
        row.update(compare_distributions(calib_vals, test_vals))
        results.append(row)

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    significant = result_df[result_df["ks_pvalue"] < 0.05]
    print(f"\nSo dac trung co drift co Y NGHIA THONG KE (KS test p<0.05): "
          f"{len(significant)}/{len(result_df)}")
    if len(significant) > 0:
        print("Danh sach:", ", ".join(significant["feature"].tolist()))
        print("\n=> Ket luan: co bang chung TRUC TIEP rang exchangeability giua tap calibration "
              "va test bi vi pham - giai thich vi sao Conformal Prediction kem hieu qua hon tren "
              "OPSD so voi GEFCom2014 (da quan sat gian tiep qua Ablation B, epistemic/aleatoric, "
              "va multi-seed robustness truoc do).")

    result_df.to_csv(os.path.join(RESULTS_DIR, "opsd_drift_analysis.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for data, label, color in [(calib_load, "Calibration window", "tab:blue"),
                                (test_load, "Test window", "tab:orange")]:
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[0].plot(sorted_data, ecdf, label=label, color=color)
    axes[0].set_xlabel("LOAD")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("So sanh ECDF: Calibration vs Test window (OPSD)")
    axes[0].legend()

    axes[1].hist(calib_load, bins=50, alpha=0.5, label="Calibration window", color="tab:blue", density=True)
    axes[1].hist(test_load, bins=50, alpha=0.5, label="Test window", color="tab:orange", density=True)
    axes[1].set_xlabel("LOAD")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Histogram: Calibration vs Test window (OPSD)")
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "opsd_calibration_vs_test_distribution.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nDa luu figure: {fig_path}")

    return result_df


if __name__ == "__main__":
    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]

    run_drift_analysis(opsd, opsd_features)

    print("\nHoan tat - ket qua luu trong results/tables/opsd_drift_analysis.csv va "
          "figure trong results/figures/opsd_calibration_vs_test_distribution.png. "
          "Dung cho Section 5.3-5.5 (giai thich nguyen nhan conformal kem hieu qua tren OPSD).")