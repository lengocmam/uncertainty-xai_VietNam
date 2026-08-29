"""
Kiểm định thống kê trên kết quả multi-seed ĐÃ CÓ SẴN (không cần chạy lại
huấn luyện) - đọc từ results/tables/multiseed_raw_*.csv.

Câu hỏi kiểm định: "Coverage SAU khi hiệu chỉnh conformal có khác biệt CÓ
Ý NGHĨA THỐNG KÊ so với coverage TRƯỚC hiệu chỉnh không, hay chênh lệch đó
chỉ là ngẫu nhiên?" Dùng 2 kiểm định song song (paired, vì cùng 1 seed cho
ra 1 cặp raw/calibrated):
  - Paired t-test: giả định phân phối chuẩn của hiệu số
  - Wilcoxon signed-rank test: không giả định phân phối, phù hợp mẫu nhỏ

Đo trên 2 đại lượng:
  1. Coverage thô (raw vs calibrated) - khác biệt có ý nghĩa không
  2. Độ lệch tuyệt đối so với mục tiêu 90% (|coverage - 0.9|) - đây mới là
     câu hỏi thực sự quan trọng: hiệu chỉnh có làm coverage GẦN mục tiêu
     hơn một cách có ý nghĩa thống kê không, không chỉ đơn thuần "khác nhau"
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join("results", "tables")
TARGET_COVERAGE = 0.90


def run_tests_for(name: str):
    path = os.path.join(RESULTS_DIR, f"multiseed_raw_{name}.csv")
    df = pd.read_csv(path)
    n = len(df)

    cov_raw = df["coverage_raw"].to_numpy()
    cov_cal = df["coverage_calibrated"].to_numpy()

    print(f"\n{'='*60}\nKIỂM ĐỊNH THỐNG KÊ — {name} (n={n} seed)\n{'='*60}")

    # ----- Kiểm định 1: coverage_raw vs coverage_calibrated -----
    t_stat, p_ttest = stats.ttest_rel(cov_raw, cov_cal)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(cov_raw, cov_cal)
    except ValueError as e:
        w_stat, p_wilcoxon = np.nan, np.nan
        print(f"  [Wilcoxon lỗi: {e} - có thể do mẫu quá nhỏ hoặc hiệu số bằng 0]")

    print(f"\n  So sánh coverage_raw vs coverage_calibrated:")
    print(f"    Paired t-test : t={t_stat:.3f}, p={p_ttest:.4f}")
    print(f"    Wilcoxon      : W={w_stat:.3f}, p={p_wilcoxon:.4f}")

    # ----- Kiểm định 2: độ lệch tuyệt đối so với mục tiêu 90% -----
    dev_raw = np.abs(cov_raw - TARGET_COVERAGE)
    dev_cal = np.abs(cov_cal - TARGET_COVERAGE)
    t_stat2, p_ttest2 = stats.ttest_rel(dev_raw, dev_cal)
    try:
        w_stat2, p_wilcoxon2 = stats.wilcoxon(dev_raw, dev_cal, alternative="greater")
    except ValueError as e:
        w_stat2, p_wilcoxon2 = np.nan, np.nan
        print(f"  [Wilcoxon (độ lệch) lỗi: {e}]")

    print(f"\n  So sánh ĐỘ LỆCH so với mục tiêu 90% (raw vs calibrated, "
          f"kỳ vọng dev_raw > dev_calibrated):")
    print(f"    Mean |dev| raw        : {dev_raw.mean():.4f}")
    print(f"    Mean |dev| calibrated : {dev_cal.mean():.4f}")
    print(f"    Paired t-test : t={t_stat2:.3f}, p={p_ttest2:.4f}")
    print(f"    Wilcoxon (1-phía): W={w_stat2:.3f}, p={p_wilcoxon2:.4f}")

    # ----- Cảnh báo về cỡ mẫu -----
    min_possible_p = 1 / (2 ** n)  # p nhỏ nhất Wilcoxon có thể đạt với n mẫu
    print(f"\n  LƯU Ý QUAN TRỌNG: với n={n} seed, Wilcoxon signed-rank test "
          f"chỉ có thể đạt p tối thiểu = {min_possible_p:.4f} dù MỌI cặp đều "
          f"cùng chiều - {'ĐỦ' if min_possible_p < 0.05 else 'KHÔNG ĐỦ'} để "
          f"đạt ngưỡng ý nghĩa thống kê thông thường (p<0.05) chỉ vì cỡ mẫu nhỏ. "
          f"{'Nên tăng số seed lên >=10 nếu muốn kết luận chắc chắn hơn.' if min_possible_p >= 0.05 else ''}")

    return {
        "dataset": name,
        "n_seeds": n,
        "coverage_ttest_p": p_ttest,
        "coverage_wilcoxon_p": p_wilcoxon,
        "deviation_ttest_p": p_ttest2,
        "deviation_wilcoxon_p": p_wilcoxon2,
        "mean_dev_raw": dev_raw.mean(),
        "mean_dev_calibrated": dev_cal.mean(),
    }


if __name__ == "__main__":
    results = []
    for name in ["gefcom2014", "opsd"]:
        try:
            results.append(run_tests_for(name))
        except FileNotFoundError:
            print(f"[Bỏ qua {name}] Không tìm thấy multiseed_raw_{name}.csv - "
                  f"cần chạy multi_seed_robustness.py trước.")

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(os.path.join(RESULTS_DIR, "statistical_significance_summary.csv"), index=False)
        print(f"\n{'='*60}\nTÓM TẮT\n{'='*60}")
        print(summary.to_string(index=False))
        print("\nHoàn tất - kết quả lưu trong results/tables/statistical_significance_summary.csv")