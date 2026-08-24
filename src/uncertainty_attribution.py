"""
Module ĐÓNG GÓP CHÍNH của bài báo: Temporal Uncertainty Attribution.

Trả lời câu hỏi: "Khoảng dự báo (conformal interval) rộng là do ĐẶC TRƯNG NÀO,
tại THỜI ĐIỂM (lag) NÀO gây ra, và bao nhiêu phần trăm là do mô hình chưa đủ
hiểu (epistemic) so với do nhiễu vốn có trong dữ liệu (aleatoric)?"

4 bước, đúng theo Methodology:
  1. Conformal calibration — hiệu chỉnh khoảng dự báo để đảm bảo coverage
  2. Temporal uncertainty attribution — mở rộng ý tưởng proportional Shapley
     (Idrissi et al., 2025) sang chiều thời gian: báo cáo đóng góp theo
     TỪNG LAG riêng biệt, không gộp chung "load" như cách làm cũ.
  3. Epistemic vs Aleatoric decomposition — qua ensemble bootstrap
  4. Stability metric — Kendall's W qua nhiều lần chạy attribution

Chạy sau khi đã có train_baseline.py chạy được (dùng lại cùng dữ liệu Gold).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import kendalltau

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUANTILES = [0.05, 0.5, 0.95]
N_BOOTSTRAP = 8          # số mô hình ensemble cho epistemic uncertainty
N_PERMUTATION_REPEATS = 5  # số lần lặp permutation cho mỗi đặc trưng (giảm nhiễu ước lượng)


# ============================================================
# BƯỚC 1 — Conformal Calibration (split conformal, đơn giản, đúng chuẩn)
# ============================================================
def conformal_calibrate(y_calib, lower_calib, upper_calib, alpha=0.1):
    """
    Split conformal prediction: tính "nonconformity score" trên tập
    calibration, lấy phân vị (1-alpha) để mở rộng/thu hẹp khoảng dự báo
    sao cho ĐẢM BẢO coverage (1-alpha), không chỉ "hy vọng" như quantile
    regression thô.
    """
    scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
    q_level = np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)
    q_level = min(q_level, 1.0)
    correction = np.quantile(scores, q_level)
    return correction  # cộng/trừ giá trị này vào lower/upper để hiệu chỉnh


# ============================================================
# BƯỚC 2 — Temporal Uncertainty Attribution (ĐÓNG GÓP CHÍNH)
# ============================================================
def interval_width_given_models(model_lower, model_upper, X):
    lower = model_lower.predict(X)
    upper = model_upper.predict(X)
    return np.mean(upper - lower)


def temporal_uncertainty_attribution(model_lower, model_upper, X_test, feature_cols, seed=0):
    """
    Permutation-based attribution cho ĐỘ RỘNG của conformal interval
    (không phải cho giá trị dự báo) — đây là bản triển khai thực nghiệm,
    mở rộng ý tưởng proportional Shapley của Idrissi et al. (2025):
    thay vì coi mỗi lag là 1 đặc trưng độc lập vô nghĩa, ta BÁO CÁO RIÊNG
    từng lag (load_lag_1, load_lag_24, load_lag_168) để thấy rõ THỜI ĐIỂM
    nào trong quá khứ gây ra bất định nhiều nhất — đây chính là phần mà
    SHAP/Idrissi gốc KHÔNG làm được.

    Cách tính: hoán vị (permute) từng đặc trưng, đo độ rộng khoảng dự báo
    thay đổi bao nhiêu so với baseline -> càng thay đổi nhiều, đặc trưng đó
    càng "chịu trách nhiệm" cho độ bất định.
    """
    rng = np.random.RandomState(seed)
    baseline_width = interval_width_given_models(model_lower, model_upper, X_test)

    contributions = {}
    for i, col in enumerate(feature_cols):
        deltas = []
        for _ in range(N_PERMUTATION_REPEATS):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_width = interval_width_given_models(model_lower, model_upper, X_perm)
            deltas.append(perm_width - baseline_width)
        contributions[col] = np.mean(deltas)

    # Chuẩn hóa thành tỷ lệ % đóng góp (chỉ lấy phần dương - đặc trưng làm TĂNG bất định)
    total_positive = sum(max(v, 0) for v in contributions.values())
    contributions_pct = {
        k: (max(v, 0) / total_positive * 100 if total_positive > 0 else 0)
        for k, v in contributions.items()
    }
    return contributions, contributions_pct, baseline_width


# ============================================================
# BƯỚC 3 — Epistemic vs Aleatoric decomposition (ensemble bootstrap)
# ============================================================
def epistemic_aleatoric_decomposition(X_train, y_train, X_test, feature_cols, n_bootstrap=N_BOOTSTRAP):
    """
    - Epistemic uncertainty: độ lệch chuẩn giữa nhiều mô hình huấn luyện
      trên các bootstrap sample khác nhau -> mô hình càng "không chắc"
      giữa các lần train khác nhau, epistemic càng cao (thường giảm khi
      có thêm dữ liệu).
    - Aleatoric uncertainty: độ rộng trung bình khoảng quantile [0.05,0.95]
      của MỖI mô hình riêng lẻ -> phản ánh nhiễu vốn có trong dữ liệu,
      không giảm dù có thêm dữ liệu.
    """
    n = len(X_train)
    rng = np.random.RandomState(42)
    median_preds = []
    aleatoric_widths = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        Xb, yb = X_train[idx], y_train[idx]

        m_low = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=150,
                                           max_depth=4, random_state=b)
        m_med = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=150,
                                           max_depth=4, random_state=b)
        m_up = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=150,
                                          max_depth=4, random_state=b)
        m_low.fit(Xb, yb); m_med.fit(Xb, yb); m_up.fit(Xb, yb)

        median_preds.append(m_med.predict(X_test))
        aleatoric_widths.append(np.mean(m_up.predict(X_test) - m_low.predict(X_test)))

        print(f"    Bootstrap model {b+1}/{n_bootstrap} xong")

    median_preds = np.array(median_preds)  # shape (n_bootstrap, n_test)
    epistemic = np.std(median_preds, axis=0).mean()   # độ lệch giữa các mô hình
    aleatoric = np.mean(aleatoric_widths)              # độ rộng quantile trung bình

    total = epistemic + aleatoric
    epistemic_pct = epistemic / total * 100 if total > 0 else 0
    aleatoric_pct = aleatoric / total * 100 if total > 0 else 0
    return {"epistemic": epistemic, "aleatoric": aleatoric,
            "epistemic_pct": epistemic_pct, "aleatoric_pct": aleatoric_pct}


# ============================================================
# BƯỚC 4 — Stability metric (Kendall's W qua nhiều lần chạy attribution)
# ============================================================
def stability_kendalls_w(model_lower, model_upper, X_test, feature_cols, n_runs=5):
    """Chạy attribution nhiều lần với seed khác nhau, đo độ ổn định của
    THỨ HẠNG các đặc trưng (không phải giá trị tuyệt đối) bằng Kendall's tau
    trung bình theo cặp -> càng gần 1, phương pháp càng ổn định."""
    rankings = []
    for run in range(n_runs):
        _, pct, _ = temporal_uncertainty_attribution(model_lower, model_upper, X_test, feature_cols, seed=run)
        ranked = sorted(pct.items(), key=lambda x: -x[1])
        rankings.append([f for f, _ in ranked])

    taus = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            rank_i = {f: r for r, f in enumerate(rankings[i])}
            rank_j = {f: r for r, f in enumerate(rankings[j])}
            common = list(rank_i.keys())
            tau, _ = kendalltau([rank_i[f] for f in common], [rank_j[f] for f in common])
            taus.append(tau)
    return np.mean(taus)


# ============================================================
# ĐIỀU PHỐI — chạy trên GEFCom2014 (đổi sang OPSD tương tự nếu cần)
# ============================================================
def run_full_pipeline(name: str, df: pd.DataFrame, feature_cols: list, target_col: str = "LOAD"):
    print(f"\n{'='*60}\nTEMPORAL UNCERTAINTY ATTRIBUTION — {name}\n{'='*60}")

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

    print("  Huấn luyện mô hình quantile chính (lower/upper)...")
    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300,
                                             max_depth=4, random_state=42)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300,
                                             max_depth=4, random_state=42)
    model_lower.fit(X_train, y_train)
    model_upper.fit(X_train, y_train)

    # Bước 1: Conformal calibration
    lower_calib = model_lower.predict(X_calib)
    upper_calib = model_upper.predict(X_calib)
    correction = conformal_calibrate(y_calib, lower_calib, upper_calib, alpha=0.1)
    print(f"  Hệ số hiệu chỉnh conformal: {correction:.3f}")

    # Bước 2: Temporal uncertainty attribution
    contributions, contributions_pct, baseline_width = temporal_uncertainty_attribution(
        model_lower, model_upper, X_test, feature_cols)
    print("\n  Đóng góp của từng đặc trưng vào ĐỘ RỘNG khoảng dự báo:")
    for f, pct in sorted(contributions_pct.items(), key=lambda x: -x[1]):
        print(f"    {f:20s}: {pct:5.1f}%")

    # Bước 3: Epistemic vs Aleatoric
    print("\n  Đang tính epistemic/aleatoric decomposition (ensemble bootstrap)...")
    decomposition = epistemic_aleatoric_decomposition(X_train, y_train, X_test, feature_cols)
    print(f"  Epistemic: {decomposition['epistemic_pct']:.1f}% | "
          f"Aleatoric: {decomposition['aleatoric_pct']:.1f}%")

    # Bước 4: Stability
    print("\n  Đang tính độ ổn định (Kendall's W qua 5 lần chạy)...")
    stability = stability_kendalls_w(model_lower, model_upper, X_test, feature_cols)
    print(f"  Độ ổn định thứ hạng đặc trưng (Kendall's tau trung bình): {stability:.3f}")

    result = pd.DataFrame({
        "feature": list(contributions_pct.keys()),
        "contribution_pct": list(contributions_pct.values()),
    }).sort_values("contribution_pct", ascending=False)
    result.to_csv(os.path.join(RESULTS_DIR, f"uncertainty_attribution_{name}.csv"), index=False)

    summary = {"baseline_interval_width": baseline_width, "conformal_correction": correction,
               **decomposition, "stability_kendall_tau": stability}
    pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, f"uncertainty_summary_{name}.csv"), index=False)

    return result, summary


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_full_pipeline("gefcom2014", gefcom, gefcom_features)

    print("\nHoàn tất — kết quả lưu trong results/tables/uncertainty_attribution_*.csv "
          "và uncertainty_summary_*.csv. Đây là bảng số liệu chính cho Section 5.1-5.3 của bài báo.")