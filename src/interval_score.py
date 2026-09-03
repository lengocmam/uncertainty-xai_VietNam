"""
Interval Score (Winkler Score) - metric bat buoc de tranh viec "danh lua"
coverage bang cach lam interval qua rong. Dung truc tiep predictions_*.csv
da co, KHONG can huan luyen lai.

IS_alpha(L,U;y) = (U-L) + (2/alpha)(L-y)*1(y<L) + (2/alpha)(y-U)*1(y>U)
alpha = 0.10 (ung voi interval 90%). Cang THAP cang tot.
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join("results", "tables")
ALPHA = 0.10


def winkler_score(y_true, lower, upper, alpha=ALPHA):
    width = upper - lower
    below = (y_true < lower).astype(float)
    above = (y_true > upper).astype(float)
    penalty_below = (2 / alpha) * (lower - y_true) * below
    penalty_above = (2 / alpha) * (y_true - upper) * above
    return width + penalty_below + penalty_above


def compute_for_dataset(name: str):
    path = os.path.join(RESULTS_DIR, f"predictions_{name}.csv")
    df = pd.read_csv(path)

    df["interval_score"] = winkler_score(
        df["y_true"].to_numpy(), df["lower_bound_final"].to_numpy(), df["upper_bound_final"].to_numpy())

    summary = df.groupby(["method", "seed"])["interval_score"].mean().reset_index()
    agg = summary.groupby("method")["interval_score"].agg(["mean", "std"]).reset_index()
    agg.insert(0, "dataset", name)

    print(f"\n{'='*60}\nINTERVAL SCORE (Winkler, alpha={ALPHA}) - {name}\n{'='*60}")
    print(agg.to_string(index=False))

    summary.insert(0, "dataset", name)
    return summary, agg


if __name__ == "__main__":
    all_summary, all_agg = [], []
    for name in ["gefcom2014", "opsd"]:
        if not os.path.exists(os.path.join(RESULTS_DIR, f"predictions_{name}.csv")):
            print(f"[BO QUA {name}] khong tim thay predictions_{name}.csv")
            continue
        summary, agg = compute_for_dataset(name)
        all_summary.append(summary)
        all_agg.append(agg)

    if all_agg:
        pd.concat(all_summary, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "interval_score_per_seed.csv"), index=False)
        final = pd.concat(all_agg, ignore_index=True)
        final.to_csv(os.path.join(RESULTS_DIR, "interval_score_summary.csv"), index=False)
        print(f"\n{'='*60}\nBANG TONG HOP (mean +/- std qua 5 seed)\n{'='*60}")
        for _, row in final.iterrows():
            print(f"  {row['dataset']:12s} {row['method']:15s}: "
                  f"{row['mean']:.3f} +/- {row['std']:.3f}")
        print("\nHoan tat - luu trong results/tables/interval_score_summary.csv va "
              "interval_score_per_seed.csv. Score cang THAP cang tot - dung de bo sung "
              "vao Bang chinh Section 5.1, chung minh interval khong 'gian lan' bang cach rong qua muc.")