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


# ============================================================
# BASELINE BỔ SUNG 1: MC Dropout (Gal & Ghahramani, 2016)
# ============================================================
class _MCDropoutNet(nn.Module):
    """Kiến trúc GIỐNG _GRUQuantileNet nhưng có thêm Dropout - dropout được
    GIỮ BẬT ngay cả lúc dự báo (test-time dropout), mỗi lần forward cho ra
    1 kết quả hơi khác nhau -> nhiều lần forward tạo thành 1 phân phối xấp
    xỉ posterior, dùng để ước lượng khoảng bất định."""

    def __init__(self, n_static_features, hidden_size=16, dropout_p=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + n_static_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 1),
        )

    def forward(self, lag_seq, static_feats):
        _, h_n = self.gru(lag_seq)
        h_last = self.dropout1(h_n.squeeze(0))
        combined = torch.cat([h_last, static_feats], dim=1)
        return self.fc(combined).squeeze(-1)


class MCDropoutRegressor:
    """
    Huấn luyện MỘT mạng duy nhất (dự báo điểm, loss MSE) có Dropout.
    Khi dự báo, chạy T lần forward VỚI DROPOUT VẪN BẬT (không gọi .eval()),
    lấy phân vị 5%/95% của T lần đó làm khoảng dự báo - đây chính là cách
    MC Dropout ước lượng bất định mà KHÔNG cần huấn luyện riêng cho từng
    quantile như GRUQuantileRegressor.
    """

    def __init__(self, n_lag_features=3, hidden_size=16, dropout_p=0.2,
                 epochs=200, lr=3e-3, batch_size=256, n_mc_samples=50,
                 random_state=42, verbose=False):
        self.n_lag_features = n_lag_features
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.x_mean_ = self.x_std_ = self.y_mean_ = self.y_std_ = None

    def _split(self, X):
        return X[:, :self.n_lag_features], X[:, self.n_lag_features:]

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.x_mean_, self.x_std_ = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-8
        X_norm = (X - self.x_mean_) / self.x_std_
        y_norm = (y - self.y_mean_) / self.y_std_

        lag, static = self._split(X_norm)
        self.model = _MCDropoutNet(static.shape[1], self.hidden_size, self.dropout_p)

        lag_t = torch.tensor(lag).unsqueeze(-1)
        static_t = torch.tensor(static)
        y_t = torch.tensor(y_norm)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = len(y_t)

        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                pred = self.model(lag_t[idx], static_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if self.verbose and (epoch + 1) % 40 == 0:
                print(f"      [MCDropout] epoch {epoch+1}/{self.epochs} - MSE (chuẩn hóa): {total_loss/n:.4f}")
        return self

    def predict_interval(self, X, lower_q=0.05, upper_q=0.95):
        """Trả về (lower, median, upper) qua T lần forward CÓ dropout."""
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.x_mean_) / self.x_std_
        lag, static = self._split(X_norm)
        lag_t = torch.tensor(lag).unsqueeze(-1)
        static_t = torch.tensor(static)

        self.model.train()  # QUAN TRỌNG: giữ dropout bật, không gọi .eval()
        samples = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                pred_norm = self.model(lag_t, static_t).numpy()
                samples.append(pred_norm * self.y_std_ + self.y_mean_)
        samples = np.stack(samples, axis=0)  # (n_mc_samples, N)

        lower = np.quantile(samples, lower_q, axis=0)
        median = np.quantile(samples, 0.5, axis=0)
        upper = np.quantile(samples, upper_q, axis=0)
        return lower, median, upper


# ============================================================
# BASELINE BỔ SUNG 2: Deep Ensembles (Lakshminarayanan et al., 2017)
# ============================================================
class DeepEnsembleQuantileRegressor:
    """
    Huấn luyện M mô hình GRUQuantileRegressor ĐỘC LẬP (khác random_state,
    CÙNG một tập train đầy đủ - KHÔNG bootstrap resample, đây là điểm khác
    biệt với epistemic_aleatoric_decomposition() trong uncertainty_attribution.py
    vốn dùng bootstrap) cho mỗi quantile, rồi lấy TRUNG BÌNH dự báo của cả
    M mô hình - đây chính là ý tưởng cốt lõi của Deep Ensembles: đa dạng
    hóa qua khởi tạo ngẫu nhiên khác nhau, không phải qua lấy mẫu dữ liệu.
    """

    def __init__(self, alpha, n_members=5, n_lag_features=3, epochs=150,
                 lr=3e-3, verbose=False):
        self.alpha = alpha
        self.n_members = n_members
        self.n_lag_features = n_lag_features
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.members = []

    def fit(self, X, y):
        self.members = []
        for m in range(self.n_members):
            model = GRUQuantileRegressor(
                alpha=self.alpha, n_lag_features=self.n_lag_features,
                epochs=self.epochs, lr=self.lr, random_state=1000 + m,  # seed khác nhau, KHÔNG bootstrap
                verbose=False)
            model.fit(X, y)  # dùng NGUYÊN tập train, không resample
            self.members.append(model)
            if self.verbose:
                print(f"      [DeepEnsemble alpha={self.alpha}] thành viên {m+1}/{self.n_members} xong")
        return self

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.members], axis=0)
        return preds.mean(axis=0)  # trung bình qua M thành viên