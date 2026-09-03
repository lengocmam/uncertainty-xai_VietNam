"""
Audit integrity cho 2 file predictions_gefcom2014.csv va predictions_opsd.csv
truoc khi viet Results - day KHONG phai them thi nghiem, chi la kiem tra
tinh nhat quan cua du lieu da co.

Kiem tra:
  1. Tong so dong theo dataset x method x seed
  2. Ty le NaN
  3. So timestamp trung (duplicate) trong tung nhom dataset/method/seed
  4. lower_bound_final <= upper_bound_final (moi dong)
  5. interval_width == upper_bound_final - lower_bound_final (sai so nho)
  6. covered_90 tinh lai tu raw columns co khop cot da luu khong
  7. Coverage tinh lai tu file predictions co khop voi multiseed_method_comparison_*.csv khong
  8. Timestamp test co GIONG HET NHAU giua 3 method, cung dataset/seed khong
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join("results", "tables")
TOLERANCE = 1e-6


def audit_dataset(name: str):
    print(f"\n{'='*70}\nAUDIT — {name}\n{'='*70}")
    path = os.path.join(RESULTS_DIR, f"predictions_{name}.csv")
    if not os.path.exists(path):
        print(f"  [BO QUA] Khong tim thay {path}")
        return None

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    issues = []

    # ----- 1. Tong so dong theo method x seed -----
    print("\n1. So dong theo method x seed:")
    counts = df.groupby(["method", "seed"]).size().unstack()
    print(counts.to_string())
    if counts.to_numpy().flatten().tolist().count(counts.to_numpy().flat[0]) != counts.size:
        issues.append("So dong KHONG dong nhat giua cac method/seed - kiem tra lai split.")

    # ----- 2. Ty le NaN -----
    print("\n2. Ty le NaN theo cot:")
    nan_ratio = df.isna().mean()
    print(nan_ratio[nan_ratio > 0].to_string() if nan_ratio.sum() > 0 else "  Khong co NaN nao.")
    if nan_ratio.sum() > 0:
        issues.append(f"Co NaN trong du lieu: {nan_ratio[nan_ratio>0].to_dict()}")

    # ----- 3. Timestamp trung trong tung nhom -----
    print("\n3. Kiem tra timestamp trung lap trong tung nhom (method, seed):")
    dup_total = 0
    for (method, seed), g in df.groupby(["method", "seed"]):
        n_dup = g["timestamp"].duplicated().sum()
        dup_total += n_dup
        if n_dup > 0:
            print(f"  [CANH BAO] method={method}, seed={seed}: {n_dup} timestamp trung")
    if dup_total == 0:
        print("  Khong co timestamp trung lap trong bat ky nhom nao.")
    else:
        issues.append(f"Tong {dup_total} timestamp trung lap phat hien.")

    # ----- 4. lower_bound_final <= upper_bound_final -----
    print("\n4. Kiem tra lower_bound_final <= upper_bound_final:")
    bad_bounds = (df["lower_bound_final"] > df["upper_bound_final"]).sum()
    print(f"  So dong VI PHAM (lower > upper): {bad_bounds} / {len(df)}")
    if bad_bounds > 0:
        issues.append(f"{bad_bounds} dong co lower_bound_final > upper_bound_final.")

    # ----- 5. interval_width == upper - lower -----
    print("\n5. Kiem tra interval_width == upper_bound_final - lower_bound_final:")
    recomputed_width = df["upper_bound_final"] - df["lower_bound_final"]
    width_diff = (df["interval_width"] - recomputed_width).abs()
    bad_width = (width_diff > TOLERANCE).sum()
    print(f"  So dong lech qua nguong {TOLERANCE}: {bad_width} / {len(df)} "
          f"(max lech = {width_diff.max():.2e})")
    if bad_width > 0:
        issues.append(f"{bad_width} dong co interval_width khong khop upper-lower.")

    # ----- 6. covered_90 tinh lai co khop cot da luu -----
    print("\n6. Kiem tra covered_90 (da luu) vs tinh lai tu raw columns:")
    recomputed_cov = ((df["y_true"] >= df["lower_bound_final"]) &
                       (df["y_true"] <= df["upper_bound_final"])).astype(int)
    mismatch = (df["covered_90"] != recomputed_cov).sum()
    print(f"  So dong khong khop: {mismatch} / {len(df)}")
    if mismatch > 0:
        issues.append(f"{mismatch} dong co covered_90 khong khop khi tinh lai.")

    # ----- 7. Coverage tinh tu predictions vs multiseed_method_comparison -----
    print("\n7. Doi chieu coverage tinh tu predictions.csv vs multiseed_method_comparison.csv:")
    ms_path = os.path.join(RESULTS_DIR, f"multiseed_method_comparison_{name}.csv")
    method_map = {"GRU+CP": "main", "MCDropout": "mcdropout", "DeepEnsemble": "deepensemble"}
    if os.path.exists(ms_path):
        ms = pd.read_csv(ms_path)
        max_diff = 0.0
        for method_pred, method_col in method_map.items():
            for seed in df["seed"].unique():
                sub = df[(df["method"] == method_pred) & (df["seed"] == seed)]
                cov_from_pred = sub["covered_90"].mean()
                cov_from_ms = ms.loc[ms["seed"] == seed, f"{method_col}_coverage"]
                if len(cov_from_ms) == 0:
                    continue
                diff = abs(cov_from_pred - cov_from_ms.iloc[0])
                max_diff = max(max_diff, diff)
        print(f"  Chenh lech LON NHAT giua 2 nguon: {max_diff:.6f}")
        if max_diff > 1e-4:
            issues.append(f"Coverage giua predictions.csv va multiseed_method_comparison.csv "
                          f"lech toi {max_diff:.6f} - can kiem tra lai 2 script co dung chung "
                          f"logic khong.")
    else:
        print(f"  [BO QUA] Khong tim thay {ms_path}")

    # ----- 8. Timestamp giong het nhau giua 3 method, cung seed -----
    print("\n8. Kiem tra timestamp GIONG HET giua 3 method (cung seed):")
    all_match = True
    for seed in df["seed"].unique():
        ts_by_method = {}
        for method in df["method"].unique():
            sub = df[(df["method"] == method) & (df["seed"] == seed)].sort_values("timestamp")
            ts_by_method[method] = sub["timestamp"].reset_index(drop=True)
        methods_list = list(ts_by_method.keys())
        base = ts_by_method[methods_list[0]]
        for m in methods_list[1:]:
            if not base.equals(ts_by_method[m]):
                all_match = False
                print(f"  [CANH BAO] seed={seed}: timestamp cua {m} KHONG khop voi {methods_list[0]}")
    if all_match:
        print("  Timestamp khop hoan toan giua 3 method, cho moi seed.")
    else:
        issues.append("Timestamp KHONG dong nhat giua cac method cung seed.")

    # ----- Tong ket -----
    print(f"\n{'-'*70}")
    if len(issues) == 0:
        print(f"KET QUA: PASS - khong phat hien van de nao. Du lieu nhat quan, "
              f"co the tin tuong bang so lieu va figure duoc xay tu file nay.")
    else:
        print(f"KET QUA: CO {len(issues)} VAN DE CAN KIEM TRA LAI:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    return issues


if __name__ == "__main__":
    all_issues = {}
    for name in ["gefcom2014", "opsd"]:
        issues = audit_dataset(name)
        all_issues[name] = issues

    print(f"\n{'='*70}\nTOM TAT CUOI CUNG\n{'='*70}")
    total_issues = sum(len(v) for v in all_issues.values() if v)
    if total_issues == 0:
        print("TAT CA DATASET DEU PASS - du lieu san sang de viet Results/Figures chinh thuc.")
    else:
        print(f"TONG CONG {total_issues} van de can xu ly truoc khi viet Results.")