import os
os.system('cls')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===================== 1) 基本设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

LAMBDA = 2 * np.pi
TOTAL_PERIODS = 4
TOTAL_STEPS = 800
SEQ_LEN = 32  # Transformer通常需要稍长窗口（可调）

# ======== 手动修改这里即可：0.25 / 0.5 / 1.0 ========
TRAIN_FRAC_LAMBDA = 1
# ================================================

# 训练参数
EPOCHS = 800
LR = 2e-3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 256
DROPOUT = 0.1

# ===================== 2) 生成数据 =====================
x_all = np.linspace(0, TOTAL_PERIODS * LAMBDA, TOTAL_STEPS)
y_all = np.sin(x_all).astype(np.float32)

# ===================== 3) 构造数据集（滑窗：用过去SEQ_LEN预测下一个） =====================
def create_dataset(y, seq_len):
    X, Y = [], []
    for i in range(len(y) - seq_len):
        X.append(y[i:i + seq_len])     # (seq_len,)
        Y.append(y[i + seq_len])       # scalar
    X = np.array(X, dtype=np.float32)[:, :, None]  # (N, seq_len, 1)
    Y = np.array(Y, dtype=np.float32)[:, None]     # (N, 1)
    return X, Y

# ===================== 4) Positional Encoding =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ===================== 5) Transformer 预测器（Encoder-only + causal mask） =====================
class TransformerForecaster(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=4096)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(T, device):
        # True表示masked（不可见）；上三角为True -> 只看过去
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask  # (T,T)

    def forward(self, x):
        # x: (B,T,1)
        B, T, _ = x.shape
        h = self.in_proj(x)             # (B,T,D)
        h = self.pos_enc(h)             # (B,T,D)
        mask = self.causal_mask(T, x.device)
        z = self.encoder(h, mask=mask)  # (B,T,D)
        y = self.out_proj(z[:, -1, :])  # 用最后一步表征预测下一个点 -> (B,1)
        return y

# ===================== 6) 自回归外推 =====================
@torch.no_grad()
def autoregressive_predict(model, y_init, total_steps, seq_len):
    model.eval()
    seq = list(y_init.astype(np.float32))
    while len(seq) < total_steps:
        x_input = np.array(seq[-seq_len:], dtype=np.float32)[None, :, None]  # (1, T, 1)
        x_tensor = torch.from_numpy(x_input).to(device)
        y_next = model(x_tensor).cpu().numpy()[0, 0]
        seq.append(float(y_next))
    return np.array(seq, dtype=np.float32)

# ===================== 7) 切分训练长度并训练 =====================
train_len = int(TOTAL_STEPS * TRAIN_FRAC_LAMBDA / TOTAL_PERIODS)
train_len = max(train_len, SEQ_LEN + 1)
y_train = y_all[:train_len]

X_train, Y_train = create_dataset(y_train, SEQ_LEN)
X_train = torch.from_numpy(X_train).to(device)
Y_train = torch.from_numpy(Y_train).to(device)

model = TransformerForecaster(
    d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

loss_history = []
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.6f}")

# ===================== 8) 外推预测 =====================
y_init = y_all[:SEQ_LEN]
y_pred = autoregressive_predict(model, y_init, TOTAL_STEPS, SEQ_LEN)

# ===================== 9) 画图（只输出两张） =====================
# 图1：Loss - Epoch
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(f"Transformer Training Loss (train length = {TRAIN_FRAC_LAMBDA} λ)")
plt.grid(True)
plt.show()

# 图2：拟合 + 外推
plt.figure(figsize=(8, 5))
plt.plot(x_all, y_all, label="True sin(x)")
plt.plot(x_all, y_pred, "--", label="Transformer prediction")

plt.axvspan(0, TRAIN_FRAC_LAMBDA * LAMBDA, alpha=0.2, label="Training region")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Transformer Extrapolation of sin(x)\n(train length = {TRAIN_FRAC_LAMBDA} λ)")
plt.legend()
plt.grid(True)
plt.show()
