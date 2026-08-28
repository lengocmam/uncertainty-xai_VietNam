"""Test nhanh: kiểm tra GRUQuantileRegressor chạy được từ đầu đến cuối
(fit + predict) trên dữ liệu giả lập, trước khi gắn vào pipeline thật."""

import numpy as np
from gru_quantile_model import GRUQuantileRegressor

rng = np.random.RandomState(0)
n = 2000

# Giả lập: 3 cột lag (tương quan với nhau) + 3 cột tĩnh
lag_1 = rng.normal(100, 10, n)
lag_24 = lag_1 + rng.normal(0, 5, n)
lag_168 = lag_1 + rng.normal(0, 8, n)
hour = rng.randint(0, 24, n).astype(float)
dayofweek = rng.randint(0, 7, n).astype(float)
temp = rng.normal(20, 5, n)

X = np.column_stack([lag_1, lag_24, lag_168, hour, dayofweek, temp])
y = 0.5 * lag_1 + 0.3 * lag_24 + 2 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 3, n)

print("Huấn luyện thử mô hình GRU (quantile 0.5, 30 epochs)...")
model = GRUQuantileRegressor(alpha=0.5, epochs=30, verbose=True)
model.fit(X[:1600], y[:1600])

pred = model.predict(X[1600:])
mae = np.mean(np.abs(pred - y[1600:]))
print(f"\nMAE trên tập test giả lập: {mae:.3f} (giá trị y dao động quanh {y.mean():.1f})")
print("Nếu MAE nhỏ hơn nhiều so với độ lệch chuẩn của y, mô hình đã học được — GRU hoạt động đúng.")
print(f"Độ lệch chuẩn của y: {y.std():.3f}")