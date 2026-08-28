"""
Mô hình GRU Quantile Regression, bọc lại theo interface giống sklearn
(.fit(X, y) / .predict(X)) để CÓ THỂ THAY THẾ TRỰC TIẾP GradientBoostingRegressor
ở mọi nơi khác (train_baseline.py, shap_library_attribution.py, ablation_study.py)
mà KHÔNG cần sửa code ở những file đó.

Thiết kế: 3 đặc trưng lag (load_lag_1, load_lag_24, load_lag_168) được coi là
MỘT CHUỖI THỜI GIAN thật sự (đưa qua GRU để học quan hệ tuần tự giữa chúng),
các đặc trưng tĩnh còn lại (hour, dayofweek, temp/solar/wind...) được ghép
vào sau qua 1 lớp fully-connected. Nhờ vậy:
  - Vẫn dùng đúng GRU (đáp ứng kỳ vọng reviewer về kiến trúc time series)
  - Vẫn giữ NGUYÊN tên từng đặc trưng (load_lag_1, load_lag_24...) để
    SHAP/attribution ở các bước sau tiếp tục giải thích được TỪNG đặc trưng
    riêng biệt, không bị "nuốt" vào 1 khối chuỗi vô danh.

QUY ƯỚC BẮT BUỘC: 3 cột lag PHẢI là 3 cột ĐẦU TIÊN trong feature_cols,
theo đúng thứ tự [lag_1, lag_24, lag_168] (hoặc thứ tự lag tăng dần).
Các cột còn lại là đặc trưng tĩnh.
"""

import numpy as np
import torch
import torch.nn as nn


class _GRUQuantileNet(nn.Module):
    def __init__(self, n_static_features, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + n_static_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, lag_seq, static_feats):
        # lag_seq: (batch, seq_len=3, 1) ; static_feats: (batch, n_static)
        _, h_n = self.gru(lag_seq)          # h_n: (1, batch, hidden_size)
        h_last = h_n.squeeze(0)             # (batch, hidden_size)
        combined = torch.cat([h_last, static_feats], dim=1)
        return self.fc(combined).squeeze(-1)


def pinball_loss_torch(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1) * diff))


class GRUQuantileRegressor:
    """
    Interface giống sklearn: .fit(X, y) và .predict(X).
    X: mảng 2D (n_samples, n_features) - 3 CỘT ĐẦU là lag, còn lại là tĩnh.
    """

    def __init__(self, alpha=0.5, n_lag_features=3, hidden_size=16,
                 epochs=200, lr=3e-3, batch_size=256, random_state=42, verbose=False):
        self.alpha = alpha
        self.n_lag_features = n_lag_features
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.y_std_ = None

    def _split(self, X):
        lag = X[:, :self.n_lag_features]
        static = X[:, self.n_lag_features:]
        return lag, static

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.x_mean_) / self.x_std_

        # QUAN TRỌNG: chuẩn hóa cả y - nếu không, mạng neural học rất chậm
        # hoặc không hội tụ khi y có thang đo lớn (ví dụ hàng trăm MW).
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-8
        y_norm = (y - self.y_mean_) / self.y_std_

        lag, static = self._split(X_norm)
        n_static = static.shape[1]
        self.model = _GRUQuantileNet(n_static, self.hidden_size)

        lag_t = torch.tensor(lag).unsqueeze(-1)
        static_t = torch.tensor(static)
        y_t = torch.tensor(y_norm)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(y_t)

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                pred = self.model(lag_t[idx], static_t[idx])
                loss = pinball_loss_torch(y_t[idx], pred, self.alpha)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"      [GRU alpha={self.alpha}] epoch {epoch+1}/{self.epochs} "
                      f"- pinball loss (đã chuẩn hóa): {total_loss/n:.4f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.x_mean_) / self.x_std_
        lag, static = self._split(X_norm)
        lag_t = torch.tensor(lag).unsqueeze(-1)
        static_t = torch.tensor(static)
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(lag_t, static_t).numpy()
        self.model.train()
        # Đưa dự báo về lại thang đo gốc của y
        return pred_norm * self.y_std_ + self.y_mean_