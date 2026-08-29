"""
Thay thế bản Shapley TỰ VIẾT (đã gặp lỗi bất ổn định) bằng thư viện `shap`
đã được kiểm chứng rộng rãi — cách tiếp cận này AN TOÀN HƠN để trình bày
trước reviewer, vì `shap` là thư viện chuẩn, được hàng chục nghìn bài báo
trích dẫn, đã xử lý sẵn các vấn đề về lấy mẫu nền, độ ổn định số học.

Ý tưởng: định nghĩa một "mô hình" mà đầu ra là ĐỘ RỘNG khoảng dự báo
(không phải giá trị dự báo) — đây chính là ý tưởng của Idrissi et al.
(2025), giờ được cài đặt qua `shap.KernelExplainer` thay vì tự viết
công thức tổ hợp.

Cài đặt thêm: pip install shap
"""

import os
# QUAN TRỌNG: phải đặt TRƯỚC khi import numpy/torch/sklearn - sửa lỗi
# xung đột thư viện OpenMP/MKL giữa torch và scikit-learn trên Windows.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import shap
from gru_quantile_model import GRUQuantileRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BACKGROUND_SUMMARY = 50   # số điểm đại diện cho nền (shap.kmeans nén lại, ổn định hơn random)
N_TEST_EXPLAIN = 200        # số điểm test đem giải thích (giảm nếu máy chậm)


def interval_width_predict_fn(model_lower, model_upper):
    """Bọc 2 mô hình quantile thành 1 hàm duy nhất trả về ĐỘ RỘNG khoảng
    dự báo cho mỗi dòng input — đây chính là "model" mà shap sẽ giải thích."""
    def predict_fn(X):
        lower = model_lower.predict(X)
        upper = model_upper.predict(X)
        return upper - lower
    return predict_fn


def run_shap_for(name: str, df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    print(f"\n{'='*60}\nSHAP (LIBRARY) UNCERTAINTY ATTRIBUTION — {name}\n{'='*60}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    train_df, test_df = df.iloc[:train_end], df.iloc[train_end:]

    X_train = train_df[feature_cols].astype("float64").to_numpy()
    y_train = train_df[target_col].astype("float64").to_numpy()
    X_test = test_df[feature_cols].astype("float64").to_numpy()

    print("  Huấn luyện mô hình quantile (lower/upper) — GRU...")
    model_lower = GRUQuantileRegressor(alpha=0.05, n_lag_features=3, epochs=200, lr=3e-3, verbose=True)
    model_upper = GRUQuantileRegressor(alpha=0.95, n_lag_features=3, epochs=200, lr=3e-3, verbose=True)
    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)

    predict_fn = interval_width_predict_fn(model_lower, model_upper)

    # Nén tập nền bằng k-means (shap.kmeans) - ổn định hơn nhiều so với
    # lấy ngẫu nhiên hoặc tự viết KNN, đây là cách shap khuyến nghị chính thức.
    print(f"  Nén tập nền còn {N_BACKGROUND_SUMMARY} điểm đại diện (shap.kmeans)...")
    background_summary = shap.kmeans(X_train, N_BACKGROUND_SUMMARY)

    print(f"  Khởi tạo KernelExplainer và tính SHAP values cho {N_TEST_EXPLAIN} điểm test "
          f"(có thể mất vài phút)...")
    explainer = shap.KernelExplainer(predict_fn, background_summary)

    rng = np.random.RandomState(0)
    test_idx = rng.choice(len(X_test), size=min(N_TEST_EXPLAIN, len(X_test)), replace=False)
    X_test_subset = X_test[test_idx]

    shap_values = explainer.shap_values(X_test_subset, nsamples="auto")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # SỬA LỖI QUAN TRỌNG: dùng mean |SHAP| (trị tuyệt đối) làm thước đo
    # đóng góp toàn cục - ĐÚNG chuẩn của Lundberg & Lee (2017). Trước đó
    # dùng mean SHAP CÓ DẤU rồi cắt bỏ số âm là SAI: một đặc trưng làm
    # khoảng dự báo RỘNG RA ở một số điểm và HẸP LẠI ở điểm khác sẽ bị
    # triệt tiêu về gần 0 khi lấy trung bình có dấu, dù nó thực sự ảnh
    # hưởng mạnh (thể hiện đúng qua |SHAP| lớn).
    mean_signed_shap = shap_values.mean(axis=0)
    total_abs = mean_abs_shap.sum()
    pct = mean_abs_shap / total_abs * 100 if total_abs > 0 else mean_abs_shap

    print("\n  Kết quả SHAP (thư viện chuẩn) — % đóng góp vào độ rộng khoảng dự báo:")
    for f, p, v in sorted(zip(feature_cols, pct, mean_signed_shap), key=lambda x: -x[1]):
        print(f"    {f:20s}: {p:5.1f}%  (mean SHAP = {v:.4f}, mean |SHAP| = {mean_abs_shap[feature_cols.index(f)]:.4f})")

    result = pd.DataFrame({
        "feature": feature_cols,
        "mean_shap_value": mean_signed_shap,
        "mean_abs_shap": mean_abs_shap,
        "contribution_pct": pct,
    }).sort_values("contribution_pct", ascending=False)
    result.to_csv(os.path.join(RESULTS_DIR, f"shap_library_{name}.csv"), index=False)
    return result


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_shap_for("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    run_shap_for("opsd", opsd, opsd_features)

    print("\nHoàn tất — kết quả lưu trong results/tables/shap_library_*.csv. "
          "Dùng bản này làm kết quả CHÍNH THỨC cho Section 5 của bài báo "
          "(đáng tin cậy hơn bản tự viết vì dùng thư viện đã kiểm chứng).")