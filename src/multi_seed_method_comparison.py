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


import time

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

    # ----- Phương pháp chính: GRU (3 quantile THẬT: 0.05, 0.5, 0.95) + Conformal -----
    t0 = time.time()
    m_lo = GRUQuantileRegressor(alpha=0.05, n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    m_med = GRUQuantileRegressor(alpha=0.5, n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    m_hi = GRUQuantileRegressor(alpha=0.95, n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    m_lo.fit(X_train, y_train); m_med.fit(X_train, y_train); m_hi.fit(X_train, y_train)
    row["main_train_time_s"] = time.time() - t0

    lo_calib, hi_calib = m_lo.predict(X_calib), m_hi.predict(X_calib)
    correction = conformal_correction(y_calib, lo_calib, hi_calib)
    lo_test = m_lo.predict(X_test) - correction
    hi_test = m_hi.predict(X_test) + correction
    med_test = m_med.predict(X_test)  # median THẬT, không xấp xỉ (lo+hi)/2

    row["main_pinball_0.05"] = pinball_loss(y_test, lo_test, 0.05)
    row["main_pinball_0.5"] = pinball_loss(y_test, med_test, 0.5)
    row["main_pinball_0.95"] = pinball_loss(y_test, hi_test, 0.95)
    row["main_coverage"] = coverage(y_test, lo_test, hi_test)
    row["main_calibration_error"] = abs(row["main_coverage"] - TARGET_COVERAGE)
    row["main_width"] = np.mean(hi_test - lo_test)

    # ----- MC Dropout (median THẬT từ phân vị 50% của các lần lấy mẫu MC) -----
    t0 = time.time()
    mc = MCDropoutRegressor(n_lag_features=3, epochs=EPOCHS_ROBUST, random_state=seed)
    mc.fit(X_train, y_train)
    row["mcdropout_train_time_s"] = time.time() - t0
    lo_mc, med_mc, hi_mc = mc.predict_interval(X_test)
    row["mcdropout_pinball_0.05"] = pinball_loss(y_test, lo_mc, 0.05)
    row["mcdropout_pinball_0.5"] = pinball_loss(y_test, med_mc, 0.5)
    row["mcdropout_pinball_0.95"] = pinball_loss(y_test, hi_mc, 0.95)
    row["mcdropout_coverage"] = coverage(y_test, lo_mc, hi_mc)
    row["mcdropout_calibration_error"] = abs(row["mcdropout_coverage"] - TARGET_COVERAGE)
    row["mcdropout_width"] = np.mean(hi_mc - lo_mc)

    # ----- Deep Ensembles (3 quantile THẬT: 0.05, 0.5, 0.95, mỗi quantile là 1 ensemble riêng) -----
    t0 = time.time()
    de_lo = DeepEnsembleQuantileRegressor(alpha=0.05, n_members=N_ENSEMBLE_MEMBERS,
                                           epochs=EPOCHS_ROBUST, random_state=seed)
    de_med = DeepEnsembleQuantileRegressor(alpha=0.5, n_members=N_ENSEMBLE_MEMBERS,
                                            epochs=EPOCHS_ROBUST, random_state=seed)
    de_hi = DeepEnsembleQuantileRegressor(alpha=0.95, n_members=N_ENSEMBLE_MEMBERS,
                                           epochs=EPOCHS_ROBUST, random_state=seed)
    de_lo.fit(X_train, y_train); de_med.fit(X_train, y_train); de_hi.fit(X_train, y_train)
    row["deepensemble_train_time_s"] = time.time() - t0

    lo_de, med_de, hi_de = de_lo.predict(X_test), de_med.predict(X_test), de_hi.predict(X_test)
    row["deepensemble_pinball_0.05"] = pinball_loss(y_test, lo_de, 0.05)
    row["deepensemble_pinball_0.5"] = pinball_loss(y_test, med_de, 0.5)
    row["deepensemble_pinball_0.95"] = pinball_loss(y_test, hi_de, 0.95)
    row["deepensemble_coverage"] = coverage(y_test, lo_de, hi_de)
    row["deepensemble_calibration_error"] = abs(row["deepensemble_coverage"] - TARGET_COVERAGE)
    row["deepensemble_width"] = np.mean(hi_de - lo_de)

    # ----- Prediction-level output: 1 dòng / timestamp / phương pháp -----
    # Dùng chung timestamps cho cả 3 phương pháp (cùng test_df) - đúng yêu cầu.
    timestamps = test_df["timestamp"].to_numpy()
    pred_blocks = []

    # main: q05_raw/q95_raw là dự báo GRU TRƯỚC hiệu chỉnh conformal;
    # lower/upper_bound_final là SAU hiệu chỉnh. q50 không bị conformal điều chỉnh.
    lo_raw_main = m_lo.predict(X_test)
    hi_raw_main = m_hi.predict(X_test)
    pred_blocks.append(pd.DataFrame({
        "seed": seed, "method": "GRU+CP", "timestamp": timestamps, "y_true": y_test,
        "q05_raw": lo_raw_main, "q50": med_test, "q95_raw": hi_raw_main,
        "lower_bound_final": lo_test, "upper_bound_final": hi_test,
        "interval_width": hi_test - lo_test,
        "covered_90": ((y_test >= lo_test) & (y_test <= hi_test)).astype(int),
    }))

    # MC Dropout và Deep Ensemble không có bước hiệu chỉnh riêng biệt trong
    # nghiên cứu này -> raw == final (ghi rõ để không hiểu nhầm là đã calibrate).
    pred_blocks.append(pd.DataFrame({
        "seed": seed, "method": "MCDropout", "timestamp": timestamps, "y_true": y_test,
        "q05_raw": lo_mc, "q50": med_mc, "q95_raw": hi_mc,
        "lower_bound_final": lo_mc, "upper_bound_final": hi_mc,
        "interval_width": hi_mc - lo_mc,
        "covered_90": ((y_test >= lo_mc) & (y_test <= hi_mc)).astype(int),
    }))
    pred_blocks.append(pd.DataFrame({
        "seed": seed, "method": "DeepEnsemble", "timestamp": timestamps, "y_true": y_test,
        "q05_raw": lo_de, "q50": med_de, "q95_raw": hi_de,
        "lower_bound_final": lo_de, "upper_bound_final": hi_de,
        "interval_width": hi_de - lo_de,
        "covered_90": ((y_test >= lo_de) & (y_test <= hi_de)).astype(int),
    }))
    predictions_df = pd.concat(pred_blocks, ignore_index=True)

    return row, predictions_df


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


def build_official_table(all_results: dict):
    """all_results: {dataset_name: results_df} -> bảng chính thức mean±std,
    đúng định dạng: Dataset | Method | Pinball(0.05/0.5/0.95) | Coverage |
    Calibration error | Mean width | Training cost."""
    method_labels = {
        "main": ("GRU + CP", f"1x GRU x3 quantile ({EPOCHS_ROBUST} epoch)"),
        "mcdropout": ("MC Dropout", f"1x GRU ({EPOCHS_ROBUST} epoch) + 50 MC forward passes"),
        "deepensemble": ("Deep Ensemble", f"{N_ENSEMBLE_MEMBERS}x GRU x3 quantile ({EPOCHS_ROBUST} epoch)"),
    }
    metrics = ["pinball_0.05", "pinball_0.5", "pinball_0.95", "coverage",
               "calibration_error", "width", "train_time_s"]

    rows = []
    for dataset, df in all_results.items():
        for method_key, (method_name, cost_label) in method_labels.items():
            r = {"Dataset": dataset, "Method": method_name, "Training cost": cost_label}
            for metric in metrics:
                col = f"{method_key}_{metric}"
                mean_v, std_v = df[col].mean(), df[col].std()
                if metric == "train_time_s":
                    r["Train time (s)"] = f"{mean_v:.1f} +/- {std_v:.1f}"
                else:
                    label = {"pinball_0.05": "Pinball q=0.05", "pinball_0.5": "Pinball q=0.50",
                             "pinball_0.95": "Pinball q=0.95", "coverage": "90% Coverage",
                             "calibration_error": "Calibration error", "width": "Mean width"}[metric]
                    r[label] = f"{mean_v:.3f} +/- {std_v:.3f}"
            rows.append(r)

    table = pd.DataFrame(rows)
    return table


def run_all_for(name: str, df: pd.DataFrame, feature_cols: list):
    print(f"\n{'='*60}\nMULTI-SEED METHOD COMPARISON - {name} ({N_SEEDS} seeds)\n{'='*60}")
    rows = []
    all_predictions = []
    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        row, predictions_df = run_one_seed(df, feature_cols, seed)
        rows.append(row)
        predictions_df.insert(0, "dataset", name)
        all_predictions.append(predictions_df)
        print(f"    main: cov={row['main_coverage']:.3f} pb={row['main_pinball_0.5']:.3f} | "
              f"MCDropout: cov={row['mcdropout_coverage']:.3f} pb={row['mcdropout_pinball_0.5']:.3f} | "
              f"DeepEns: cov={row['deepensemble_coverage']:.3f} pb={row['deepensemble_pinball_0.5']:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv"), index=False)

    predictions_all_df = pd.concat(all_predictions, ignore_index=True)
    pred_path = os.path.join(RESULTS_DIR, f"predictions_{name}.csv")
    predictions_all_df.to_csv(pred_path, index=False)
    print(f"\n  Da luu prediction-level output: {pred_path} ({len(predictions_all_df):,} dong "
          f"= {N_SEEDS} seed x 3 method x {len(predictions_all_df)//(N_SEEDS*3):,} timestamp)")

    tests_df = run_significance_tests(results_df, name)
    tests_df.to_csv(os.path.join(RESULTS_DIR, f"significance_method_comparison_{name}.csv"), index=False)
    return results_df, tests_df


if __name__ == "__main__":
    print("PROTOCOL CHÍNH THỨC (dùng DUY NHẤT bảng này cho bài báo, KHÔNG trộn với "
          "baseline_*.csv cũ vốn dùng split 80/20 không calibration và 200 epoch):")
    print(f"  - Split: 60% train / 20% calibration / 20% test (theo thời gian, không xáo trộn)")
    print(f"  - Epoch: {EPOCHS_ROBUST} cho CẢ 3 phương pháp (đồng nhất)")
    print(f"  - Deep Ensemble: {N_ENSEMBLE_MEMBERS} thành viên/quantile")
    print(f"  - Số seed lặp lại: {N_SEEDS}\n")

    all_results = {}

    gefcom = pd.read_parquet(os.path.join(GOLD_DIR, "gefcom2014_features.parquet"))
    gefcom_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek", "temp_best"]
    results_gefcom, _ = run_all_for("gefcom2014", gefcom, gefcom_features)
    all_results["gefcom2014"] = results_gefcom

    opsd = pd.read_parquet(os.path.join(GOLD_DIR, "opsd_features.parquet"))
    opsd_features = ["load_lag_1", "load_lag_24", "load_lag_168", "hour", "dayofweek"]
    opsd_features += [c for c in ["solar", "wind"] if c in opsd.columns]
    results_opsd, _ = run_all_for("opsd", opsd, opsd_features)
    all_results["opsd"] = results_opsd

    official_table = build_official_table(all_results)
    official_table.to_csv(os.path.join(RESULTS_DIR, "official_results_table.csv"), index=False)
    print(f"\n{'='*60}\nBANG CHINH THUC (mean +/- std qua {N_SEEDS} seed)\n{'='*60}")
    print(official_table.to_string(index=False))

    print("\nHoan tat - bang chinh thuc luu trong results/tables/official_results_table.csv. "
          "Day la SO LIEU DUY NHAT nen dung cho Section 5.1 cua bai bao - "
          "khong trich lai baseline_*.csv (script cu, khac protocol).")