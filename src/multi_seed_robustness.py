"""
Kiểm định độ ổn định thống kê (statistical robustness) — chạy lại toàn bộ
quy trình huấn luyện GRU với NHIỀU SEED khác nhau, báo cáo "mean ± std"
thay vì 1 con số đơn — gần như bắt buộc với journal Q1/Q2 để chứng minh
kết quả không phải ngẫu nhiên may mắn từ 1 lần chạy.

Chạy SAU khi đã có Gold data (pipeline_datasets.py) và gru_quantile_model.py.
CẢNH BÁO: file này chạy N_SEEDS lần toàn bộ quy trình huấn luyện GRU ->
rất chậm (N_SEEDS x thời gian 1 lần train_baseline.py). Giảm N_SEEDS nếu
cần chạy nhanh hơn để kiểm tra trước.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from gru_quantile_model import GRUQuantileRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5   # giảm xuống 3 nếu máy chậm, cần tối thiểu 5 để báo cáo std có ý nghĩa
QUANTILES = [0.05, 0.5, 0.95]


def pinball_loss(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def conformal_correction(y_calib, lower_calib, upper_calib, alpha=0.1):
    scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
    q_level = min(np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), 1.0)
    return np.quantile(scores, q_level)


def run_one_seed(df: pd.DataFrame, feature_cols: list, seed: int, target_col: str = "LOAD"):
    """1 lần chạy đầy đủ: train GRU (3 quantile) trên 1 seed, trả về metric
    CẢ TRƯỚC và SAU khi hiệu chỉnh conformal - để so sánh độ ổn định (std)
    giữa 2 giai đoạn qua nhiều seed."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end, calib_end = int(n * 0.6), int(n * 0.8)
    train_df = df.iloc[:train_end]
    calib_df = df.iloc[train_end:calib_end]
    test_df = df.iloc[calib_end:]

    X_train = train_df[feature_cols].astype("float64").to_numpy()
    y_train = train_df[target_col].astype("float64").to_numpy()
    X_calib = calib_df[feature_cols].astype("float64").to_numpy()
    y_calib = calib_df[target_col].astype("float64").to_numpy()
    X_test = test_df[feature_cols].astype("float64").to_numpy()
    y_test = test_df[target_col].astype("float64").to_numpy()

    preds_test, preds_calib = {}, {}
    for q in QUANTILES:
        model = GRUQuantileRegressor(alpha=q, n_lag_features=3, epochs=150, lr=3e-3, random_state=seed)
        model.fit(X_train, y_train)
        preds_test[q] = model.predict(X_test)
        preds_calib[q] = model.predict(X_calib)

    pinball_05 = pinball_loss(y_test, preds_test[0.05], 0.05)
    pinball_50 = pinball_loss(y_test, preds_test[0.5], 0.5)
    pinball_95 = pinball_loss(y_test, preds_test[0.95], 0.95)

    cov_raw = coverage(y_test, preds_test[0.05], preds_test[0.95])
    width_raw = np.mean(preds_test[0.95] - preds_test[0.05])

    correction = conformal_correction(y_calib, preds_calib[0.05], preds_calib[0.95])
    lower_cal = preds_test[0.05] - correction
    upper_cal = preds_test[0.95] + correction
    cov_calibrated = coverage(y_test, lower_cal, upper_cal)
    width_calibrated = np.mean(upper_cal - lower_cal)

    return {
        "seed": seed,
        "pinball_0.05": pinball_05,
        "pinball_0.5": pinball_50,
        "pinball_0.95": pinball_95,
        "coverage_raw": cov_raw,
        "interval_width_raw": width_raw,
        "coverage_calibrated": cov_calibrated,
        "interval_width_calibrated": width_calibrated,
    }


def run_multi_seed_for(name: str, df: pd.DataFrame, feature_cols: list):
    print(f"\n{'='*60}\nMULTI-SEED ROBUSTNESS CHECK — {name} ({N_SEEDS} seeds)\n{'='*60}")

    all_runs = []
    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        result = run_one_seed(df, feature_cols, seed)
        all_runs.append(result)
        print(f"    coverage_raw={result['coverage_raw']:.4f}  "
              f"coverage_calibrated={result['coverage_calibrated']:.4f}  "
              f"pinball_0.5={result['pinball_0.5']:.4f}")

    results_df = pd.DataFrame(all_runs)
    summary = results_df.drop(columns=["seed"]).agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    print(f"\n  KẾT QUẢ TỔNG HỢP QUA {N_SEEDS} SEED ({name}):")
    for metric, row in summary.iterrows():
        print(f"    {metric:25s}: {row['mean']:.4f} +/- {row['std']:.4f}")

    std_raw = summary.loc["coverage_raw", "std"]
    std_cal = summary.loc["coverage_calibrated", "std"]
    print(f"\n  SO SÁNH ĐỘ ỔN ĐỊNH COVERAGE: std TRƯỚC hiệu chỉnh = {std_raw:.4f} "
          f"-> std SAU hiệu chỉnh = {std_cal:.4f} "
          f"({'GIẢM' if std_cal < std_raw else 'KHÔNG giảm'} "
          f"{abs(std_raw-std_cal)/std_raw*100:.1f}%)")

    results_df.to_csv(os.path.join(RESULTS_DIR, f"multiseed_raw_{name}.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, f"multiseed_summary_{name}.csv"))
    return results_df, summary


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    _, gefcom_summary = run_multi_seed_for("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    _, opsd_summary = run_multi_seed_for("opsd", opsd, opsd_features)

    print("\nHoàn tất — kết quả lưu trong results/tables/multiseed_raw_*.csv và "
          "multiseed_summary_*.csv. Dùng dạng 'mean +/- std' này để báo cáo trong "
          "Section 5.1 của bài báo thay vì chỉ 1 con số đơn từ 1 lần chạy.")