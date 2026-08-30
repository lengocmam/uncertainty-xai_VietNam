"""
So sánh phương pháp chính (GRU + Conformal Calibration) với 2 baseline
uncertainty-aware bổ sung: MC Dropout và Deep Ensembles.

Cả 2 baseline TÁI DÙNG hạ tầng đã có (GRUQuantileRegressor, chuẩn hóa,
cách tính metric) - chỉ thêm 2 class mới trong gru_quantile_model.py.

Chạy sau khi đã có Gold data. Kết quả so sánh lưu vào
results/tables/baseline_comparison_extra.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from gru_quantile_model import GRUQuantileRegressor, MCDropoutRegressor, DeepEnsembleQuantileRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)


def pinball_loss(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def interval_width(lower, upper):
    return np.mean(upper - lower)


def evaluate(name, y_test, lower, median, upper, results):
    pb_lo = pinball_loss(y_test, lower, 0.05)
    pb_hi = pinball_loss(y_test, upper, 0.95)
    pb_med = pinball_loss(y_test, median, 0.5)
    cov = coverage(y_test, lower, upper)
    width = interval_width(lower, upper)
    print(f"    {name:20s}: coverage={cov:.4f}  width={width:.3f}  "
          f"pinball(0.05/0.5/0.95)={pb_lo:.3f}/{pb_med:.3f}/{pb_hi:.3f}")
    results.append({"method": name, "coverage": cov, "interval_width": width,
                     "pinball_0.05": pb_lo, "pinball_0.5": pb_med, "pinball_0.95": pb_hi})


def run_comparison_for(name: str, df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    print(f"\n{'='*60}\nSO SÁNH BASELINE UNCERTAINTY-AWARE — {name}\n{'='*60}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train_df[feature_cols].astype("float64").to_numpy()
    y_train = train_df[target_col].astype("float64").to_numpy()
    X_test = test_df[feature_cols].astype("float64").to_numpy()
    y_test = test_df[target_col].astype("float64").to_numpy()

    results = []

    # ----- Baseline 1: MC Dropout -----
    print("\n  Huấn luyện MC Dropout (1 mô hình, dropout p=0.2, 50 lần lấy mẫu khi dự báo)...")
    mc_model = MCDropoutRegressor(n_lag_features=3, dropout_p=0.2, epochs=200,
                                   n_mc_samples=50, verbose=True)
    mc_model.fit(X_train, y_train)
    lower_mc, median_mc, upper_mc = mc_model.predict_interval(X_test)
    evaluate("MC Dropout", y_test, lower_mc, median_mc, upper_mc, results)

    # ----- Baseline 2: Deep Ensembles -----
    print("\n  Huấn luyện Deep Ensembles (5 thành viên x 2 quantile = 10 mô hình GRU)...")
    de_lower = DeepEnsembleQuantileRegressor(alpha=0.05, n_members=5, epochs=150, verbose=True)
    de_upper = DeepEnsembleQuantileRegressor(alpha=0.95, n_members=5, epochs=150, verbose=True)
    de_lower.fit(X_train, y_train)
    de_upper.fit(X_train, y_train)
    lower_de = de_lower.predict(X_test)
    upper_de = de_upper.predict(X_test)
    median_de = (lower_de + upper_de) / 2  # xấp xỉ, Deep Ensemble ở đây không train riêng median
    evaluate("Deep Ensembles", y_test, lower_de, median_de, upper_de, results)

    # ----- Đối chiếu: phương pháp chính (đọc lại từ kết quả đã lưu trước đó) -----
    main_path = os.path.join(RESULTS_DIR, f"baseline_{name}.csv")
    if os.path.exists(main_path):
        main_df = pd.read_csv(main_path)
        print(f"\n  [Tham khảo] Phương pháp chính (GRU + Conformal, đã lưu trước đó) "
              f"tại {main_path}:")
        print(main_df.to_string(index=False))
    else:
        print(f"\n  [Lưu ý] Không tìm thấy {main_path} để đối chiếu - "
              f"chạy train_baseline.py trước nếu muốn so sánh trực tiếp.")

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(RESULTS_DIR, f"baseline_comparison_extra_{name}.csv"), index=False)
    return result_df


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_comparison_for("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    run_comparison_for("opsd", opsd, opsd_features)

    print("\nHoàn tất — kết quả lưu trong results/tables/baseline_comparison_extra_*.csv. "
          "Dùng bảng này ghép với baseline_*.csv (phương pháp chính) để làm Bảng so sánh "
          "SOTA trong Section 5.1 của bài báo.")