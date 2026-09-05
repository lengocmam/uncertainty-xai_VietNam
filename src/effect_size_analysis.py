"""
Effect size + confidence interval cho ket qua da co (5 seed) - bo sung vao
statistical_tests.py de p-value khong phai la con so duy nhat bao cao.

Tinh cho main vs mcdropout va main vs deepensemble, tren pinball_0.5 va
calibration_error (tu multiseed_method_comparison_*.csv), CONG THEM
bootstrap CI cho interval score o muc prediction-level (tu predictions_*.csv,
dung seed trung vi) - manh hon ve mat thong ke vi n lon hon nhieu so voi 5 seed.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join("results", "tables")
N_BOOTSTRAP = 2000
RNG = np.random.RandomState(42)


def cohens_dz(diff: np.ndarray) -> float:
    """Cohen's d cho paired sample (dz) = mean(diff) / std(diff)."""
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def paired_mean_ci(diff: np.ndarray, confidence=0.95):
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se


def seed_level_effect_size(name: str):
    ms = pd.read_csv(os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv"))
    rows = []
    for baseline, col in [("mcdropout", "mcdropout"), ("deepensemble", "deepensemble")]:
        for metric in ["pinball_0.5", "calibration_error"]:
            diff = (ms[f"main_{metric}"] - ms[f"{col}_{metric}"]).to_numpy()
            dz = cohens_dz(diff)
            mean_d, lo, hi = paired_mean_ci(diff)
            rows.append({
                "dataset": name, "comparison": f"main_vs_{baseline}", "metric": metric,
                "mean_paired_diff": mean_d, "ci95_lower": lo, "ci95_upper": hi,
                "cohens_dz": dz, "n_seeds": len(diff),
            })
    return pd.DataFrame(rows)


def moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Tra ve 1 bo chi so length n, ghep tu cac KHOI LIEN TIEP (moving block
    bootstrap, Kunsch 1989) thay vi lay tung diem doc lap (i.i.d. bootstrap).

    LY DO CAN THIET (diem 2 - gop y quan trong): du lieu phu tai theo GIO
    co tu tuong quan manh (gio ke nhau giong nhau) - i.i.d. bootstrap gia
    dinh sai cac diem doc lap, lam KHOANG TIN CAY BI HEP GIA TAO (qua tu
    tin). Block bootstrap giu nguyen cau truc phu thuoc trong tung khoi,
    cho uoc luong do bat dinh dung dan hon.
    """
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size
    starts = rng.randint(0, max_start + 1, size=n_blocks_needed)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return idx[:n]


def bootstrap_interval_score_ci(name: str, n_bootstrap=N_BOOTSTRAP, block_size: int = 24):
    """Block bootstrap CI (khong phai i.i.d.) cho chenh lech Interval Score
    giua main va tung baseline, o MUC PREDICTION-LEVEL. block_size=24 (1 ngay)
    mac dinh de bao toan chu ky ngay-dem trong du lieu phu tai theo gio -
    dieu chinh neu do phan giai du lieu khac (vi du 48 cho du lieu 30 phut)."""
    pred = pd.read_csv(os.path.join(RESULTS_DIR, f"predictions_{name}.csv"))
    ms = pd.read_csv(os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv"))
    median_cov = ms["main_coverage"].median()
    seed = int(ms.loc[(ms["main_coverage"] - median_cov).abs().idxmin(), "seed"])

    def winkler(y, lo, hi, alpha=0.10):
        w = hi - lo
        below = (y < lo).astype(float)
        above = (y > hi).astype(float)
        return w + (2/alpha)*(lo-y)*below + (2/alpha)*(y-hi)*above

    rows = []
    main_sub = pred[(pred["method"] == "GRU+CP") & (pred["seed"] == seed)].sort_values("timestamp")
    main_is = winkler(main_sub["y_true"].to_numpy(), main_sub["lower_bound_final"].to_numpy(),
                       main_sub["upper_bound_final"].to_numpy())

    for baseline_label, method_name in [("mcdropout", "MCDropout"), ("deepensemble", "DeepEnsemble")]:
        base_sub = pred[(pred["method"] == method_name) & (pred["seed"] == seed)].sort_values("timestamp")
        base_is = winkler(base_sub["y_true"].to_numpy(), base_sub["lower_bound_final"].to_numpy(),
                           base_sub["upper_bound_final"].to_numpy())

        n = len(main_is)
        assert n == len(base_is), "So diem test khong khop giua main va baseline!"
        diffs = main_is - base_is  # am nghia la main TOT hon (Winkler thap hon)

        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = moving_block_bootstrap_indices(n, block_size, RNG)
            boot_means[b] = diffs[idx].mean()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        rows.append({
            "dataset": name, "comparison": f"main_vs_{baseline_label}",
            "seed_used": seed, "n_test_points": n, "block_size": block_size,
            "bootstrap_method": "moving_block",
            "mean_interval_score_diff": diffs.mean(),
            "bootstrap_ci95_lower": ci_lo, "bootstrap_ci95_upper": ci_hi,
            "significant": not (ci_lo <= 0 <= ci_hi),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_seedlevel, all_bootstrap = [], []
    for name in ["gefcom2014", "opsd"]:
        ms_path = os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv")
        if not os.path.exists(ms_path):
            print(f"[BO QUA {name}] khong tim thay {ms_path}")
            continue

        print(f"\n{'='*60}\nEFFECT SIZE (seed-level, n=5) - {name}\n{'='*60}")
        seedlevel = seed_level_effect_size(name)
        print(seedlevel.to_string(index=False))
        all_seedlevel.append(seedlevel)

        print(f"\n{'='*60}\nBOOTSTRAP CI (prediction-level, seed trung vi) - {name}\n{'='*60}")
        boot = bootstrap_interval_score_ci(name)
        print(boot.to_string(index=False))
        all_bootstrap.append(boot)

    if all_seedlevel:
        pd.concat(all_seedlevel, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "effect_size_seedlevel.csv"), index=False)
        pd.concat(all_bootstrap, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "effect_size_bootstrap_interval_score.csv"), index=False)
        print("\nHoan tat - luu trong results/tables/effect_size_seedlevel.csv va "
              "effect_size_bootstrap_interval_score.csv. CI prediction-level dung MOVING "
              "BLOCK BOOTSTRAP (block_size=24 gio, khong phai i.i.d.) de ton trong tu tuong "
              "quan chuoi thoi gian - ghi ro dieu nay trong Methodology/Section 5.1, khong "
              "goi la 'bootstrap' chung chung. Dung ca 2 bang khi viet: seed-level (n=5, "
              "yeu ve power) DE BO SUNG cho block-bootstrap prediction-level (n lon hon "
              "nhieu, nhung van co the con mot phan tu tuong quan chua duoc loai bo hoan "
              "toan tuy block_size - neu trong Limitations).")