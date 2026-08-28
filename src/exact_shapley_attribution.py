"""
Shapley Value THẬT cho Temporal Uncertainty Attribution — thay thế bản
permutation-based xấp xỉ trong uncertainty_attribution.py.

CÔNG THỨC (Shapley value cổ điển, dùng cho conformal interval width):

    phi_i = sum_{S subset N\{i}}  [|S|! (|N|-|S|-1)! / |N|!] * [v(S U {i}) - v(S)]

    v(S) = độ rộng khoảng dự báo trung bình khi CHỈ các đặc trưng trong S
           giữ giá trị thật (của instance đang xét), các đặc trưng ngoài S
           bị thay bằng giá trị nền lấy mẫu từ tập train (mô phỏng "che"
           thông tin - đúng cách SHAP/Shapley định nghĩa "coalition value"
           cho dữ liệu liên tục, gọi là "interventional/marginal SHAP").

Vì |N| chỉ 6-8 đặc trưng trong bài toán này, số tổ hợp con 2^|N| = 64-256
là HOÀN TOÀN khả thi để liệt kê đầy đủ (exact Shapley), không cần xấp xỉ
Monte Carlo như KernelSHAP phải làm với hàng trăm đặc trưng.

Đây chính là cách "proportional Shapley value" mà Idrissi et al. (2025)
dùng để gán độ rộng conformal interval - áp dụng ĐÚNG công thức của họ,
nhưng mở rộng để feature set bao gồm các LAG riêng biệt (temporal), thay
vì chỉ đặc trưng tĩnh như bài gốc.
"""

import os
import itertools
from math import factorial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BACKGROUND_POOL = 2000  # kích thước tập nền để tìm K-nearest-neighbor
K_NEIGHBORS = 30          # số hàng xóm gần nhất dùng làm nền có điều kiện
N_TEST_SUBSAMPLE = 300


# ============================================================
# Hàm giá trị liên minh v(S) — bản CONDITIONAL (KNN), không còn marginal ngẫu nhiên
# ============================================================
def build_knn_structures(background_scaled: np.ndarray, n_features: int):
    """
    Với MỖI tổ hợp con S có thể có (2^n tổ hợp), xây sẵn 1 cấu trúc
    NearestNeighbors chỉ dùng các chiều thuộc S để đo khoảng cách.
    Tổ hợp rỗng (S = {}) không cần NN vì không có gì để "điều kiện hóa".

    Xây MỘT LẦN, dùng lại cho MỌI điểm test (vì các tổ hợp S chỉ phụ thuộc
    vào chỉ số đặc trưng, không phụ thuộc dữ liệu cụ thể) - tiết kiệm rất
    nhiều thời gian so với xây lại NN cho từng điểm test.
    """
    structures = {}
    all_indices = list(range(n_features))
    for r in range(1, n_features + 1):
        for S in itertools.combinations(all_indices, r):
            nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(background_scaled)))
            nn.fit(background_scaled[:, list(S)])
            structures[S] = nn
    return structures


def coalition_value_conditional(model_lower, model_upper, x_instance_scaled, x_instance_raw,
                                 background_raw, background_scaled, knn_structures, coalition_indices):
    """
    v(S) kiểu CONDITIONAL: tìm K dòng nền có giá trị GẦN GIỐNG nhất với
    điểm đang xét Ở ĐÚNG các chiều thuộc S (đo khoảng cách sau chuẩn hóa),
    rồi lấy nguyên vẹn các dòng đó để điền phần "che" (ngoài S) - tránh
    tạo ra tổ hợp phi thực tế như bản marginal ngẫu nhiên trước đó.

    Nếu S rỗng: không có gì để điều kiện hóa -> lấy ngẫu nhiên K dòng nền
    (tương đương bản marginal cũ, nhưng chỉ áp dụng cho trường hợp này).
    """
    n_features = x_instance_raw.shape[0]
    mask = np.zeros(n_features, dtype=bool)

    if len(coalition_indices) == 0:
        neighbor_idx = np.random.RandomState(0).choice(len(background_raw), size=K_NEIGHBORS, replace=False)
    else:
        mask[list(coalition_indices)] = True
        nn = knn_structures[tuple(sorted(coalition_indices))]
        query = x_instance_scaled[list(coalition_indices)].reshape(1, -1)
        _, neighbor_idx = nn.kneighbors(query)
        neighbor_idx = neighbor_idx[0]

    neighbor_rows = background_raw[neighbor_idx]  # (K, n_features) - giữ nguyên tương quan nội bộ
    X_mix = np.tile(x_instance_raw, (len(neighbor_rows), 1))
    X_mix[:, ~mask] = neighbor_rows[:, ~mask]  # chỉ thay phần NGOÀI S, lấy từ hàng xóm THẬT

    lower = model_lower.predict(X_mix)
    upper = model_upper.predict(X_mix)
    return np.mean(upper - lower)


# ============================================================
# Shapley value CHÍNH XÁC cho 1 instance — dùng coalition value CONDITIONAL
# ============================================================
def exact_shapley_single_instance(model_lower, model_upper, x_raw, x_scaled,
                                   background_raw, background_scaled, knn_structures, n_features):
    """Tính đủ 2^n tổ hợp con, áp đúng công thức Shapley cổ điển,
    dùng coalition value CÓ ĐIỀU KIỆN (KNN) thay vì marginal ngẫu nhiên."""
    all_indices = list(range(n_features))
    phi = np.zeros(n_features)
    value_cache = {}

    def get_value(subset_tuple):
        if subset_tuple not in value_cache:
            value_cache[subset_tuple] = coalition_value_conditional(
                model_lower, model_upper, x_scaled, x_raw,
                background_raw, background_scaled, knn_structures, subset_tuple)
        return value_cache[subset_tuple]

    n = n_features
    for i in all_indices:
        others = [j for j in all_indices if j != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S_set = tuple(sorted(S))
                S_with_i = tuple(sorted(S + (i,)))

                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                marginal_contribution = get_value(S_with_i) - get_value(S_set)
                phi[i] += weight * marginal_contribution

    return phi


# ============================================================
# Shapley value TOÀN CỤC — trung bình qua nhiều test instance
# ============================================================
def global_shapley_attribution(model_lower, model_upper, X_train, X_test, feature_cols, seed=0):
    """
    Tính Shapley value trung bình qua N_TEST_SUBSAMPLE điểm test, dùng
    coalition value CÓ ĐIỀU KIỆN (KNN) để tránh tổ hợp phi thực tế khi
    các đặc trưng (đặc biệt các lag) tương quan mạnh với nhau.
    """
    rng = np.random.RandomState(seed)
    n_features = len(feature_cols)

    bg_idx = rng.choice(len(X_train), size=min(N_BACKGROUND_POOL, len(X_train)), replace=False)
    background_raw = X_train[bg_idx]

    scaler = StandardScaler().fit(X_train)
    background_scaled = scaler.transform(background_raw)

    print(f"    Đang xây {2**n_features - 1} cấu trúc KNN (mỗi tổ hợp con 1 lần, dùng chung cho mọi điểm test)...")
    knn_structures = build_knn_structures(background_scaled, n_features)

    test_idx = rng.choice(len(X_test), size=min(N_TEST_SUBSAMPLE, len(X_test)), replace=False)
    test_subset_raw = X_test[test_idx]
    test_subset_scaled = scaler.transform(test_subset_raw)

    all_phi = np.zeros((len(test_subset_raw), n_features))
    for k in range(len(test_subset_raw)):
        all_phi[k] = exact_shapley_single_instance(
            model_lower, model_upper, test_subset_raw[k], test_subset_scaled[k],
            background_raw, background_scaled, knn_structures, n_features)
        if (k + 1) % 50 == 0:
            print(f"    Đã tính Shapley cho {k+1}/{len(test_subset_raw)} điểm test...")

    mean_phi = all_phi.mean(axis=0)
    positive_phi = np.maximum(mean_phi, 0)
    total = positive_phi.sum()
    pct = positive_phi / total * 100 if total > 0 else positive_phi

    return dict(zip(feature_cols, mean_phi)), dict(zip(feature_cols, pct))


# ============================================================
# ĐIỀU PHỐI
# ============================================================
def run_exact_shapley_for(name: str, df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    print(f"\n{'='*60}\nEXACT SHAPLEY UNCERTAINTY ATTRIBUTION — {name}\n{'='*60}")
    print(f"  Số đặc trưng: {len(feature_cols)} -> {2**len(feature_cols)} tổ hợp con mỗi điểm test")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    train_df, test_df = df.iloc[:train_end], df.iloc[train_end:]

    X_train = train_df[feature_cols].astype("float64").to_numpy()
    y_train = train_df[target_col].astype("float64").to_numpy()
    X_test = test_df[feature_cols].astype("float64").to_numpy()

    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=200,
                                             max_depth=4, random_state=42)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=200,
                                             max_depth=4, random_state=42)
    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)

    print("  Đang tính Shapley value chính xác (có thể mất vài phút)...")
    phi, phi_pct = global_shapley_attribution(model_lower, model_upper, X_train, X_test, feature_cols)

    print("\n  Kết quả Shapley value (proportional, % đóng góp vào độ rộng khoảng dự báo):")
    for f, pct in sorted(phi_pct.items(), key=lambda x: -x[1]):
        print(f"    {f:20s}: {pct:5.1f}%  (phi = {phi[f]:.4f})")

    result = pd.DataFrame({
        "feature": list(phi.keys()),
        "shapley_value": list(phi.values()),
        "contribution_pct": list(phi_pct.values()),
    }).sort_values("contribution_pct", ascending=False)
    result.to_csv(os.path.join(RESULTS_DIR, f"exact_shapley_{name}.csv"), index=False)
    return result


if __name__ == "__main__":
    # LƯU Ý: dùng ÍT đặc trưng hơn bản permutation trước (6 -> giữ nguyên
    # được vì 2^6=64 vẫn nhanh; nếu thêm đặc trưng, chi phí tăng theo cấp
    # số nhân - cân nhắc trước khi thêm quá 8-10 đặc trưng).
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_exact_shapley_for("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    run_exact_shapley_for("opsd", opsd, opsd_features)

    print("\nHoàn tất — kết quả lưu trong results/tables/exact_shapley_*.csv. "
          "So sánh với uncertainty_attribution_*.csv (bản permutation) để viết "
          "vào Section 5.3 (so sánh 2 cách tính attribution).")