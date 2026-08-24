"""
Pipeline xử lý dữ liệu cho bài báo — CHỈ 2 nguồn: GEFCom2014 và Open Power
System Data (OPSD). Không còn phần EVN/Lakehouse (đã tách sang tiểu luận
trường, không thuộc phạm vi bài báo này).

Mỗi nguồn có pipeline riêng, ĐỘC LẬP hoàn toàn — không gộp dữ liệu thô,
chỉ so sánh kết quả sau khi đã có model/metric.

Cấu trúc thư mục kỳ vọng (xem hướng dẫn cấu trúc thư mục đi kèm):
    data_lake/bronze/...
    data_lake/silver/<nguon>_clean.parquet
    data_lake/gold/<nguon>_features.parquet
"""

import os
import pandas as pd

BASE_DIR = "data_lake"
BRONZE = os.path.join(BASE_DIR, "bronze")
SILVER = os.path.join(BASE_DIR, "silver")
GOLD = os.path.join(BASE_DIR, "gold")

for d in [BRONZE, SILVER, GOLD]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# NGUỒN 1: GEFCom2014
# ============================================================
def load_gefcom2014_all_tasks(load_dir: str) -> pd.DataFrame:
    """
    QUAN TRỌNG: Task 1 chứa khối dữ liệu lớn ban đầu (~69 tháng),
    Task 2 - Task 15 mỗi file CHỈ chứa 1 tháng dữ liệu tăng thêm.
    Phải đọc và gộp cả 15 file lại mới có đủ dữ liệu (nhiều năm).

    load_dir: đường dẫn tới thư mục "Load" chứa các thư mục con "Task 1".."Task 15"
    """
    frames = []
    for i in range(1, 16):
        task_path = os.path.join(load_dir, f"Task {i}", f"L{i}-train.csv")
        if not os.path.exists(task_path):
            print(f"  [Bỏ qua] Không tìm thấy: {task_path}")
            continue
        df_task = pd.read_csv(task_path)
        frames.append(df_task)
        print(f"  Task {i}: {df_task.shape[0]} dòng")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["ZONEID", "TIMESTAMP"])
    print(f"[GEFCom2014] Tổng sau khi gộp {len(frames)} task và loại trùng: {df_all.shape[0]} dòng")
    return df_all


def process_gefcom2014(load_dir: str) -> pd.DataFrame:
    """Bronze -> Silver -> Gold cho GEFCom2014 — đọc và gộp đủ 15 Task."""
    df = load_gefcom2014_all_tasks(load_dir)

    # Chỉnh lại format cho đúng với timestamp thật của file bạn có
    df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], format="%m%d%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Chọn trạm nhiệt độ tương quan cao nhất với LOAD
    w_cols = [c for c in df.columns if c.startswith("w")]
    corrs = df[w_cols + ["LOAD"]].corr()["LOAD"].drop("LOAD")
    best_station = corrs.abs().idxmax()
    df["temp_best"] = df[best_station]

    silver = df[["timestamp", "LOAD", "temp_best"]].copy()
    # GEFCom2014 Task 1 cung cấp lịch sử nhiệt độ DÀI HƠN lịch sử phụ tải
    # (117 tháng nhiệt độ vs 69 tháng phụ tải) để phục vụ mô phỏng kịch bản.
    # Các dòng đầu chuỗi chỉ có nhiệt độ, chưa có LOAD -> loại bỏ tường minh ở đây.
    n_before = len(silver)
    silver = silver.dropna(subset=["LOAD"])
    n_after = len(silver)
    print(f"[GEFCom2014] Loại {n_before - n_after} dòng thiếu LOAD "
          f"(giai đoạn chỉ có nhiệt độ lịch sử, trước khi có dữ liệu phụ tải)")
    silver.to_parquet(os.path.join(SILVER, "gefcom2014_clean.parquet"))

    gold = silver.copy()
    for lag in [1, 24, 168]:
        gold[f"load_lag_{lag}"] = gold["LOAD"].shift(lag)
    gold["hour"] = gold["timestamp"].dt.hour
    gold["dayofweek"] = gold["timestamp"].dt.dayofweek
    gold = gold.dropna()
    gold.to_parquet(os.path.join(GOLD, "gefcom2014_features.parquet"))

    print(f"[GEFCom2014] Silver: {silver.shape} | Gold: {gold.shape} | trạm nhiệt độ chọn: {best_station}")
    return gold


def load_gefcom2014_benchmark(benchmark_path: str) -> pd.DataFrame:
    """Đọc riêng file benchmark (99 mức quantile) — dùng để ĐỐI CHIẾU,
    KHÔNG dùng để huấn luyện."""
    df = pd.read_csv(benchmark_path)
    df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], format="%m%d%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    print(f"[GEFCom2014 benchmark] shape: {df.shape}")
    return df


# ============================================================
# NGUỒN 2: Open Power System Data
# ============================================================
def process_opsd(raw_path: str, zone_prefix: str = "DE_transnetbw") -> pd.DataFrame:
    """Bronze -> Silver -> Gold cho OPSD, chỉ lọc 1 vùng."""
    df = pd.read_csv(raw_path)

    df["timestamp"] = pd.to_datetime(df["utc_timestamp"], errors="coerce")
    load_col = f"{zone_prefix}_load_actual_entsoe_transparency"
    solar_col = f"{zone_prefix}_solar_generation_actual"
    wind_col = f"{zone_prefix}_wind_onshore_generation_actual"

    silver = df[["timestamp", load_col]].rename(columns={load_col: "LOAD"})
    for col_name, new_name in [(solar_col, "solar"), (wind_col, "wind")]:
        if col_name in df.columns:
            silver[new_name] = df[col_name]

    silver = silver.dropna(subset=["LOAD"]).sort_values("timestamp")
    silver = silver.set_index("timestamp").resample("1h").mean().reset_index()
    silver.to_parquet(os.path.join(SILVER, "opsd_clean.parquet"))

    gold = silver.copy()
    for lag in [1, 24, 168]:
        gold[f"load_lag_{lag}"] = gold["LOAD"].shift(lag)
    gold["hour"] = gold["timestamp"].dt.hour
    gold["dayofweek"] = gold["timestamp"].dt.dayofweek
    gold = gold.dropna()
    gold.to_parquet(os.path.join(GOLD, "opsd_features.parquet"))

    print(f"[OPSD] Silver: {silver.shape} | Gold: {gold.shape} | vùng: {zone_prefix}")
    return gold


# ============================================================
# ĐIỀU PHỐI
# ============================================================
if __name__ == "__main__":
    # Chỉnh lại đường dẫn cho đúng máy bạn trước khi chạy.
    # gefcom_load_dir phải trỏ tới thư mục "Load" chứa các thư mục con "Task 1".."Task 15"
    gefcom_load_dir = r"data/raw/GEFCom2014"
    gefcom_gold = process_gefcom2014(gefcom_load_dir)
    gefcom_bench = load_gefcom2014_benchmark(
        os.path.join(gefcom_load_dir, "Task 15", "L15-benchmark.csv"))
    opsd_gold = process_opsd(r"data/raw/OPSD/time_series_60min_singleindex.csv")

    print("\nHoàn tất — 2 nguồn (GEFCom2014 + OPSD) đã qua Bronze-Silver-Gold, "
          "sẵn sàng cho bước huấn luyện baseline.")