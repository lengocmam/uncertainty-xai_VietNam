"""
SHAP Uncertainty Attribution DAY DU - diem uu tien cao nhat.

Giai thich cho w_t = q_hat_0.95(x_t) - q_hat_0.05(x_t) (interval width),
SO SANH voi SHAP cho diem du bao (q_hat_0.50), qua 5 seed de danh gia
STABILITY - day la thu bien "XAI kem them uncertainty" thanh
"uncertainty-aware explainability" co the kiem chung.

Bao cao:
  1. Global mean |SHAP| cho w_t (khong phai chi 1 seed, ma mean+-std qua 5 seed)
  2. Lag-wise: load_lag_1, load_lag_24, load_lag_168 TACH RIENG
  3. So sanh ranking: SHAP(point forecast q50) vs SHAP(interval width w_t)
  4. Stability qua 5 seed (mean +- std cho tung dac trung)
  5. Figure co error bar
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gru_quantile_model import GRUQuantileRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_SEEDS = 5
EPOCHS = 100                # dong bo voi protocol chinh thuc da dung o multi_seed_method_comparison.py
N_BACKGROUND_SUMMARY = 30   # giam nhe so voi ban don-seed truoc (50) de chay noi 5 seed trong thoi gian hop ly
N_TEST_EXPLAIN = 150


def run_one_seed(df: pd.DataFrame, feature_cols: list, seed: int, target_col: str = "LOAD"):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train_df[feature_cols].astype("float64").to_numpy()
    y_train = train_df[target_col].astype("float64").to_numpy()
    X_test = test_df[feature_cols].astype("float64").to_numpy()

    m_lo = GRUQuantileRegressor(alpha=0.05, n_lag_features=3, epochs=EPOCHS, random_state=seed)
    m_med = GRUQuantileRegressor(alpha=0.5, n_lag_features=3, epochs=EPOCHS, random_state=seed)
    m_hi = GRUQuantileRegressor(alpha=0.95, n_lag_features=3, epochs=EPOCHS, random_state=seed)
    m_lo.fit(X_train, y_train); m_med.fit(X_train, y_train); m_hi.fit(X_train, y_train)

    def width_fn(X):
        return m_hi.predict(X) - m_lo.predict(X)

    def point_fn(X):
        return m_med.predict(X)

    background = shap.kmeans(X_train, N_BACKGROUND_SUMMARY)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(len(X_test), size=min(N_TEST_EXPLAIN, len(X_test)), replace=False)
    X_test_subset = X_test[test_idx]

    explainer_width = shap.KernelExplainer(width_fn, background)
    shap_width = explainer_width.shap_values(X_test_subset, nsamples="auto")
    mean_abs_width = np.abs(shap_width).mean(axis=0)

    explainer_point = shap.KernelExplainer(point_fn, background)
    shap_point = explainer_point.shap_values(X_test_subset, nsamples="auto")
    mean_abs_point = np.abs(shap_point).mean(axis=0)

    return mean_abs_width, mean_abs_point


def run_for_dataset(name: str, df: pd.DataFrame, feature_cols: list):
    print(f"\n{'='*60}\nSHAP UNCERTAINTY ATTRIBUTION (multi-seed) - {name}\n{'='*60}")

    width_records, point_records = [], []
    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        mean_abs_width, mean_abs_point = run_one_seed(df, feature_cols, seed)
        width_records.append(dict(zip(feature_cols, mean_abs_width)))
        point_records.append(dict(zip(feature_cols, mean_abs_point)))
        print(f"    Xong seed {seed}")

    width_df = pd.DataFrame(width_records)
    point_df = pd.DataFrame(point_records)

    # Chuan hoa ve % dong gop TRONG TUNG SEED roi moi lay mean+-std qua seed
    # (dung hon la lay mean tho roi moi chuan hoa 1 lan)
    width_pct = width_df.div(width_df.sum(axis=1), axis=0) * 100
    point_pct = point_df.div(point_df.sum(axis=1), axis=0) * 100

    summary = pd.DataFrame({
        "feature": feature_cols,
        "width_shap_pct_mean": width_pct.mean().values,
        "width_shap_pct_std": width_pct.std().values,
        "point_shap_pct_mean": point_pct.mean().values,
        "point_shap_pct_std": point_pct.std().values,
    })
    summary["rank_width"] = summary["width_shap_pct_mean"].rank(ascending=False).astype(int)
    summary["rank_point"] = summary["point_shap_pct_mean"].rank(ascending=False).astype(int)
    summary["rank_shift"] = summary["rank_point"] - summary["rank_width"]
    summary = summary.sort_values("width_shap_pct_mean", ascending=False)

    print(f"\n  KET QUA (mean +/- std qua {N_SEEDS} seed):")
    print(summary.to_string(index=False))

    tau, _ = pd.Series(summary["rank_width"].values).corr(
        pd.Series(summary["rank_point"].values), method="kendall"), None
    print(f"\n  Kendall's tau giua ranking SHAP(width) va SHAP(point forecast): {tau:.3f}")
    print(f"  (Gan 1: 2 ranking gan giong nhau -> dac trung quan trong cho DU BAO cung quan trong "
          f"cho DO BAT DINH. Gan 0 hoac am: 2 ranking KHAC NHAU -> day la bang chung manh nhat "
          f"cho thay giai thich uncertainty la mot BAI TOAN KHAC, khong the suy ra tu giai thich "
          f"point forecast - dung ung ho luan diem chinh cua bai bao.)")

    summary.insert(0, "dataset", name)
    summary.to_csv(os.path.join(RESULTS_DIR, f"shap_uncertainty_vs_point_{name}.csv"), index=False)

    # Figure co error bar
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(feature_cols))
    order = summary["feature"].tolist()
    w_mean = summary.set_index("feature").loc[order, "width_shap_pct_mean"]
    w_std = summary.set_index("feature").loc[order, "width_shap_pct_std"]
    p_mean = summary.set_index("feature").loc[order, "point_shap_pct_mean"]
    p_std = summary.set_index("feature").loc[order, "point_shap_pct_std"]

    bar_w = 0.35
    ax.bar(x - bar_w/2, w_mean, bar_w, yerr=w_std, capsize=4, label="SHAP - interval width", color="tab:red", alpha=0.8)
    ax.bar(x + bar_w/2, p_mean, bar_w, yerr=p_std, capsize=4, label="SHAP - point forecast (q=0.50)", color="tab:blue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP| (% of total, +/- std across 5 seeds)")
    ax.set_title(f"SHAP Attribution: Interval Width vs Point Forecast ({name})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"figure_shap_width_vs_point_{name}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\n  Da luu figure: {fig_path}")

    return summary


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_for_dataset("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    run_for_dataset("opsd", opsd, opsd_features)

    print("\nHoan tat - ket qua luu trong results/tables/shap_uncertainty_vs_point_*.csv "
          "va figure trong results/figures/figure_shap_width_vs_point_*.png. Day la bang "
          "chung CHINH cho contribution 'uncertainty-aware explainability' cua bai bao.")