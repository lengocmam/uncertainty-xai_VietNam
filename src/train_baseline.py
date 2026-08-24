"""
Huấn luyện baseline LightGBM Quantile Regression cho GEFCom2014 và OPSD.
Đây là baseline ĐẦU TIÊN — mốc số liệu để so sánh với phương pháp
uncertainty attribution sẽ phát triển ở bước sau.

Chạy sau khi đã chạy xong pipeline_datasets.py (cần có 2 file Gold parquet).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 3 mức quantile tối thiểu để có khoảng dự báo — có thể mở rộng thêm sau
QUANTILES = [0.05, 0.5, 0.95]

FEATURE_COLS_TEMPLATE = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]


def pinball_loss(y_true, y_pred, quantile):
    """Pinball loss cho 1 mức quantile — càng thấp càng tốt."""
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def coverage(y_true, lower, upper):
    """Tỷ lệ % giá trị thật nằm trong khoảng [lower, upper]."""
    return np.mean((y_true >= lower) & (y_true <= upper))


def interval_width(lower, upper):
    """Độ rộng trung bình của khoảng dự báo."""
    return np.mean(upper - lower)


def train_quantile_lgbm(df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    """Huấn luyện 1 mô hình LightGBM riêng cho mỗi mức quantile.
    Chia train/test theo thời gian (80/20), KHÔNG shuffle ngẫu nhiên -
    bắt buộc với dữ liệu chuỗi thời gian để tránh rò rỉ thông tin tương lai."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Ép kiểu tường minh về float64 thường (không phải kiểu "nullable" của
    # pandas) — tránh lỗi access violation của LightGBM trên Windows khi đọc
    # dữ liệu vừa load lại từ parquet.
    X_train = X_train.astype("float64").to_numpy()
    y_train = y_train.astype("float64").to_numpy()
    X_test = X_test.astype("float64").to_numpy()
    y_test = y_test.astype("float64").to_numpy()

    assert not np.isnan(X_train).any(), "X_train vẫn còn NaN — kiểm tra lại bước tiền xử lý"
    assert not np.isnan(y_train).any(), "y_train vẫn còn NaN — kiểm tra lại bước tiền xử lý"

    preds = {}
    for q in QUANTILES:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds[q] = model.predict(X_test)

    return y_test, preds, test_df["timestamp"].values


def evaluate_and_report(name: str, y_test, preds):
    """In và trả về bảng kết quả pinball loss + coverage + interval width."""
    print(f"\n{'='*60}\nKẾT QUẢ BASELINE — {name}\n{'='*60}")

    rows = []
    for q, y_pred in preds.items():
        pl = pinball_loss(y_test, y_pred, q)
        print(f"  Pinball loss @ quantile {q}: {pl:.4f}")
        rows.append({"quantile": q, "pinball_loss": pl})

    lower, upper = preds[min(QUANTILES)], preds[max(QUANTILES)]
    cov = coverage(y_test, lower, upper)
    width = interval_width(lower, upper)
    expected_cov = max(QUANTILES) - min(QUANTILES)

    print(f"  Coverage thực tế (khoảng [{min(QUANTILES)}, {max(QUANTILES)}]): "
          f"{cov:.2%} (kỳ vọng {expected_cov:.0%})")
    print(f"  Độ rộng khoảng dự báo trung bình: {width:.2f}")

    result_df = pd.DataFrame(rows)
    result_df.attrs["coverage"] = cov
    result_df.attrs["interval_width"] = width
    result_df.to_csv(os.path.join(RESULTS_DIR, f"baseline_{name}.csv"), index=False)
    return result_df


if __name__ == "__main__":
    # ---- GEFCom2014 ----
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = FEATURE_COLS_TEMPLATE + ["temp_best"]
    y_test, preds, _ = train_quantile_lgbm(gefcom, gefcom_features)
    evaluate_and_report("gefcom2014", y_test, preds)

    # ---- OPSD ----
    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = FEATURE_COLS_TEMPLATE + ["solar", "wind"]
    opsd_features = [c for c in opsd_features if c in opsd.columns]  # solar/wind có thể thiếu tùy vùng
    y_test, preds, _ = train_quantile_lgbm(opsd, opsd_features)
    evaluate_and_report("opsd", y_test, preds)

    print(f"\nKết quả đã lưu vào {RESULTS_DIR}/baseline_*.csv — "
          f"dùng làm mốc so sánh cho phương pháp uncertainty attribution sau này.")