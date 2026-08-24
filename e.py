import pandas as pd
silver = pd.read_parquet("data_lake/silver/gefcom2014_clean.parquet")
print("Số dòng Silver:", len(silver))
print("Số NaN từng cột:")
print(silver.isna().sum())