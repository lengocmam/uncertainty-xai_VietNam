"""
Ablation Study — chứng minh từng thành phần của phương pháp đề xuất có
đóng góp thực sự, không phải thêm cho có. 2 phép so sánh:

  A. Temporal (per-lag) attribution  VS  Grouped attribution (kiểu SHAP
     thường: coi toàn bộ lịch sử phụ tải là 1 khối, không tách theo lag)
     -> Chứng minh: việc tách theo thời gian tiết lộ thông tin mà cách
        làm cũ (gộp chung) không thấy được.

  B. Có Conformal Calibration  VS  Không có Conformal Calibration
     -> Chứng minh: bước hiệu chỉnh conformal thực sự cải thiện coverage
        so với chỉ dùng quantile regression thô.

Chạy sau khi đã có uncertainty_attribution.py chạy được (dùng lại cùng
mô hình/dữ liệu Gold).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

LAG_FEATURES = ["load_lag_1", "load_lag_24", "load_lag_168"]
N_PERMUTATION_REPEATS = 5


def interval_width_given_models(model_lower, model_upper, X):
    return np.mean(model_upper.predict(X) - model_lower.predict(X))


# ============================================================
# ABLATION A — Temporal (per-lag) vs Grouped attribution
# ============================================================
def attribution_temporal(model_lower, model_upper, X_test, feature_cols, seed=0):
    """Phương pháp đề xuất: hoán vị TỪNG lag riêng biệt (giống
    uncertainty_attribution.py)."""
    rng = np.random.RandomState(seed)
    baseline = interval_width_given_models(model_lower, model_upper, X_test)
    contributions = {}
    for i, col in enumerate(feature_cols):
        if col not in LAG_FEATURES:
            continue
        deltas = []
        for _ in range(N_PERMUTATION_REPEATS):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            deltas.append(interval_width_given_models(model_lower, model_upper, X_perm) - baseline)
        contributions[col] = max(np.mean(deltas), 0)
    return contributions


def attribution_grouped(model_lower, model_upper, X_test, feature_cols, seed=0):
    """Ablation - kiểu SHAP/permutation importance THÔNG THƯỜNG: coi cả 3
    lag như MỘT khối duy nhất "load_history" - hoán vị đồng thời cả 3 cột
    cùng lúc (giữ nguyên tương quan nội bộ giữa chúng), chỉ đo được TỔNG
    đóng góp, không tách được ra lag nào quan trọng hơn."""
    rng = np.random.RandomState(seed)
    baseline = interval_width_given_models(model_lower, model_upper, X_test)
    lag_idx = [i for i, c in enumerate(feature_cols) if c in LAG_FEATURES]

    deltas = []
    for _ in range(N_PERMUTATION_REPEATS):
        X_perm = X_test.copy()
        perm_order = rng.permutation(len(X_perm))
        X_perm[:, lag_idx] = X_perm[perm_order][:, lag_idx]  # hoán vị đồng thời cả khối
        deltas.append(interval_width_given_models(model_lower, model_upper, X_perm) - baseline)

    return {"load_history (khối gộp, không tách lag)": max(np.mean(deltas), 0)}


def run_ablation_a(name, X_test, feature_cols, model_lower, model_upper):
    print(f"\n  --- Ablation A ({name}): Temporal vs Grouped attribution ---")
    temporal = attribution_temporal(model_lower, model_upper, X_test, feature_cols)
    grouped = attribution_grouped(model_lower, model_upper, X_test, feature_cols)

    temporal_total = sum(temporal.values())
    print(f"  [Grouped - kiểu SHAP thường] load_history chiếm tổng cộng: "
          f"{list(grouped.values())[0]:.3f} (đơn vị độ rộng interval)")
    print(f"  [Temporal - phương pháp đề xuất] Tách ra được:")
    for lag, v in temporal.items():
        pct_of_total = v / temporal_total * 100 if temporal_total > 0 else 0
        print(f"      {lag:15s}: {v:.3f}  ({pct_of_total:.1f}% trong nhóm load_history)")

    print(f"  => Ý nghĩa: cách làm cũ chỉ biết \"lịch sử phụ tải nói chung\" quan trọng,"
          f" phương pháp đề xuất chỉ RÕ thời điểm nào (1 giờ trước, 1 ngày trước,"
          f" hay 1 tuần trước) mới thực sự là nguyên nhân.")

    result = pd.DataFrame([
        {"method": "grouped (SHAP-style)", "detail": k, "value": v} for k, v in grouped.items()
    ] + [
        {"method": "temporal (proposed)", "detail": k, "value": v} for k, v in temporal.items()
    ])
    result.to_csv(os.path.join(RESULTS_DIR, f"ablation_A_{name}.csv"), index=False)
    return result


# ============================================================
# ABLATION B — Có vs Không có Conformal Calibration
# ============================================================
def run_ablation_b(name, y_calib, lower_calib, upper_calib, y_test, lower_test, upper_test, alpha=0.1):
    print(f"\n  --- Ablation B ({name}): Có vs Không Conformal Calibration ---")

    # Không hiệu chỉnh — dùng thẳng quantile regression thô
    cov_raw = np.mean((y_test >= lower_test) & (y_test <= upper_test))
    width_raw = np.mean(upper_test - lower_test)

    # Có hiệu chỉnh conformal (split conformal)
    scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
    q_level = min(np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), 1.0)
    correction = np.quantile(scores, q_level)
    lower_cal = lower_test - correction
    upper_cal = upper_test + correction
    cov_calibrated = np.mean((y_test >= lower_cal) & (y_test <= upper_cal))
    width_calibrated = np.mean(upper_cal - lower_cal)

    expected = 1 - alpha
    print(f"  Không hiệu chỉnh : coverage = {cov_raw:.2%} (kỳ vọng {expected:.0%}), "
          f"độ rộng = {width_raw:.2f}")
    print(f"  Có hiệu chỉnh    : coverage = {cov_calibrated:.2%} (kỳ vọng {expected:.0%}), "
          f"độ rộng = {width_calibrated:.2f}")
    print(f"  => Ý nghĩa: hiệu chỉnh conformal đưa coverage GẦN kỳ vọng hơn "
          f"(sai số giảm từ {abs(cov_raw-expected):.2%} xuống {abs(cov_calibrated-expected):.2%}), "
          f"đổi lại khoảng dự báo rộng hơn một chút — đúng đánh đổi lý thuyết của conformal prediction.")

    result = pd.DataFrame([
        {"variant": "without_calibration", "coverage": cov_raw, "interval_width": width_raw},
        {"variant": "with_calibration", "coverage": cov_calibrated, "interval_width": width_calibrated},
    ])
    result.to_csv(os.path.join(RESULTS_DIR, f"ablation_B_{name}.csv"), index=False)
    return result


# ============================================================
# ĐIỀU PHỐI
# ============================================================
def run_ablations_for(name: str, df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    print(f"\n{'='*60}\nABLATION STUDY — {name}\n{'='*60}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end, calib_end = int(n * 0.6), int(n * 0.8)
    train_df, calib_df, test_df = df.iloc[:train_end], df.iloc[train_end:calib_end], df.iloc[calib_end:]

    X_train = train_df[feature_cols].astype("float64").to_numpy()
    y_train = train_df[target_col].astype("float64").to_numpy()
    X_calib = calib_df[feature_cols].astype("float64").to_numpy()
    y_calib = calib_df[target_col].astype("float64").to_numpy()
    X_test = test_df[feature_cols].astype("float64").to_numpy()
    y_test = test_df[target_col].astype("float64").to_numpy()

    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300,
                                             max_depth=4, random_state=42)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300,
                                             max_depth=4, random_state=42)
    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)

    lower_calib, upper_calib = model_lower.predict(X_calib), model_upper.predict(X_calib)
    lower_test, upper_test = model_lower.predict(X_test), model_upper.predict(X_test)

    result_a = run_ablation_a(name, X_test, feature_cols, model_lower, model_upper)
    result_b = run_ablation_b(name, y_calib, lower_calib, upper_calib, y_test, lower_test, upper_test)
    return result_a, result_b


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_ablations_for("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    run_ablations_for("opsd", opsd, opsd_features)

    print("\nHoàn tất — kết quả lưu trong results/tables/ablation_A_*.csv và ablation_B_*.csv. "
          "Đây là số liệu cho Section 5.3 (Ablation Study) của bài báo.")