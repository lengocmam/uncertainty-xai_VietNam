"""
Kiểm định thống kê so sánh PHƯƠNG PHÁP CHÍNH (GRU + Conformal) với 2
baseline (MC Dropout, Deep Ensembles) qua NHIỀU SEED — trả lời câu hỏi
bắt buộc cho phản biện: "phương pháp mới có thực sự tốt hơn có Ý NGHĨA
THỐNG KÊ, hay chỉ tốt hơn ở đúng 1 lần chạy may mắn?"

CẢNH BÁO QUAN TRỌNG VỀ THỜI GIAN: mỗi seed cần huấn luyện CẢ 3 phương
pháp (phương pháp chính: 2 mô hình GRU; MC Dropout: 1 mô hình; Deep
Ensembles: 2 x N_ENSEMBLE_MEMBERS mô hình) = nhiều lần huấn luyện GRU
mỗi seed/bộ dữ liệu. Để giữ thời gian chạy trong tầm kiểm soát, số epoch
và số thành viên ensemble được GIẢM so với kết quả CHÍNH THỨC đã báo cáo
(EPOCHS_ROBUST=100 thay vì 200, N_ENSEMBLE_MEMBERS=3 thay vì 5) - cần
NÊU RÕ sự khác biệt này trong phần thực nghiệm của bài báo.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from scipy import stats
from gru_quantile_model import GRUQuantileRegressor, MCDropoutRegressor, DeepEnsembleQuantileRegressor

GOLD_DIR = os.path.join("data_lake", "gold")
RESULTS_DIR = os.path.join("results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5
EPOCHS_ROBUST = 100
N_ENSEMBLE_MEMBERS = 3
TARGET_COVERAGE = 0.90


def pinball_loss(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def conformal_correction(y_calib, lower_calib, upper_calib, alpha=0.1):
    scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
    q_level = min(np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), 1.0)
    return np.quantile(scores, q_level)


def run_one_seed(df, feature_cols, seed, target_col="LOAD"):
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

    row = {"seed": seed}

    m_lo = GRUQuantileRegressor(alpha=0.05, n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    m_hi = GRUQuantileRegressor(alpha=0.95, n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    m_lo.fit(X_train, y_train); m_hi.fit(X_train, y_train)
    lo_calib, hi_calib = m_lo.predict(X_calib), m_hi.predict(X_calib)
    correction = conformal_correction(y_calib, lo_calib, hi_calib)
    lo_test, hi_test = m_lo.predict(X_test) - correction, m_hi.predict(X_test) + correction
    row["main_coverage"] = coverage(y_test, lo_test, hi_test)
    row["main_pinball_0.5"] = pinball_loss(y_test, (lo_test + hi_test) / 2, 0.5)
    row["main_width"] = np.mean(hi_test - lo_test)

    mc = MCDropoutRegressor(n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    mc.fit(X_train, y_train)
    lo_mc, med_mc, hi_mc = mc.predict_interval(X_test)
    row["mcdropout_coverage"] = coverage(y_test, lo_mc, hi_mc)
    row["mcdropout_pinball_0.5"] = pinball_loss(y_test, med_mc, 0.5)
    row["mcdropout_width"] = np.mean(hi_mc - lo_mc)

    de_lo = DeepEnsembleQuantileRegressor(alpha=0.05, n_members=N_ENSEMBLE_MEMBERS,
                                           epochs=EPOCHS_ROBUST, random_state=seed)
    de_hi = DeepEnsembleQuantileRegressor(alpha=0.95, n_members=N_ENSEMBLE_MEMBERS,
                                           epochs=EPOCHS_ROBUST, random_state=seed)
    de_lo.fit(X_train, y_train); de_hi.fit(X_train, y_train)
    lo_de, hi_de = de_lo.predict(X_test), de_hi.predict(X_test)
    row["deepensemble_coverage"] = coverage(y_test, lo_de, hi_de)
    row["deepensemble_pinball_0.5"] = pinball_loss(y_test, (lo_de + hi_de) / 2, 0.5)
    row["deepensemble_width"] = np.mean(hi_de - lo_de)

    return row


def run_significance_tests(results_df: pd.DataFrame, name: str):
    print(f"\n  KIEM DINH THONG KE - {name}: Phuong phap chinh vs tung baseline")
    test_rows = []
    for baseline in ["mcdropout", "deepensemble"]:
        main_pb = results_df["main_pinball_0.5"].to_numpy()
        base_pb = results_df[f"{baseline}_pinball_0.5"].to_numpy()
        t_pb, p_pb_t = stats.ttest_rel(main_pb, base_pb)
        try:
            w_pb, p_pb_w = stats.wilcoxon(main_pb, base_pb, alternative="less")
        except ValueError:
            w_pb, p_pb_w = np.nan, np.nan

        main_dev = np.abs(results_df["main_coverage"].to_numpy() - TARGET_COVERAGE)
        base_dev = np.abs(results_df[f"{baseline}_coverage"].to_numpy() - TARGET_COVERAGE)
        t_cov, p_cov_t = stats.ttest_rel(main_dev, base_dev)
        try:
            w_cov, p_cov_w = stats.wilcoxon(main_dev, base_dev, alternative="less")
        except ValueError:
            w_cov, p_cov_w = np.nan, np.nan

        print(f"\n  Phuong phap chinh vs {baseline}:")
        print(f"    Pinball loss @ 0.5   : main={main_pb.mean():.3f} vs {baseline}={base_pb.mean():.3f} "
              f"| t-test p={p_pb_t:.4f}, Wilcoxon p={p_pb_w:.4f}")
        print(f"    |Coverage - 90%|     : main={main_dev.mean():.4f} vs {baseline}={base_dev.mean():.4f} "
              f"| t-test p={p_cov_t:.4f}, Wilcoxon p={p_cov_w:.4f}")

        test_rows.append({
            "dataset": name, "comparison": f"main_vs_{baseline}",
            "pinball_ttest_p": p_pb_t, "pinball_wilcoxon_p": p_pb_w,
            "coverage_dev_ttest_p": p_cov_t, "coverage_dev_wilcoxon_p": p_cov_w,
        })
    return pd.DataFrame(test_rows)


def run_all_for(name: str, df: pd.DataFrame, feature_cols: list):
    print(f"\n{'='*60}\nMULTI-SEED METHOD COMPARISON - {name} ({N_SEEDS} seeds)\n{'='*60}")
    rows = []
    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        row = run_one_seed(df, feature_cols, seed)
        rows.append(row)
        print(f"    main: cov={row['main_coverage']:.3f} pb={row['main_pinball_0.5']:.3f} | "
              f"MCDropout: cov={row['mcdropout_coverage']:.3f} pb={row['mcdropout_pinball_0.5']:.3f} | "
              f"DeepEns: cov={row['deepensemble_coverage']:.3f} pb={row['deepensemble_pinball_0.5']:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv"), index=False)

    tests_df = run_significance_tests(results_df, name)
    tests_df.to_csv(os.path.join(RESULTS_DIR, f"significance_method_comparison_{name}.csv"), index=False)
    return results_df, tests_df


if __name__ == "__main__":
    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    run_all_for("gefcom2014", gefcom, gefcom_features)

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    run_all_for("opsd", opsd, opsd_features)

    print("\nHoan tat - ket qua luu trong results/tables/multiseed_method_comparison_*.csv "
          "va significance_method_comparison_*.csv. Dung cho Section 5.1 (bang so sanh SOTA "
          "kem kiem dinh y nghia thong ke) cua bai bao.")