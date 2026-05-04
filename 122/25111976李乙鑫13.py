import os
os.system('cls')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===================== 1. 基本设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# 正弦信号参数
LAMBDA = 2 * np.pi            # sin(x) 的周期
TOTAL_PERIODS = 4             # 用于测试/外推的总周期数
TOTAL_STEPS = 800             # 总采样点数
SEQ_LEN = 13                # LSTM 输入序列长度

# ======== 手动修改这里即可 ========
TRAIN_FRAC_LAMBDA = 2   # 0.25, 0.5, 1.0
# =================================

# 训练参数
EPOCHS = 800
LR = 1e-2
HIDDEN_SIZE = 32

# ===================== 2. 生成数据 =====================
x_all = np.linspace(0, TOTAL_PERIODS * LAMBDA, TOTAL_STEPS)
y_all = np.sin(x_all)

# ===================== 3. 构造数据集 =====================
def create_dataset(y, seq_len):
    X, Y = [], []
    for i in range(len(y) - seq_len):
        X.append(y[i:i + seq_len])
        Y.append(y[i + seq_len])
    X = np.array(X)[..., None]  # (N, seq_len, 1)
    Y = np.array(Y)[..., None]  # (N, 1)
    return X, Y

# ===================== 4. LSTM 模型 =====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):  # 添加 num_layers
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出层不变


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后时间步
        return out

# ===================== 5. 自回归外推 =====================
def autoregressive_predict(model, y_init, total_steps):
    model.eval()
    seq = list(y_init.astype(np.float32))

    with torch.no_grad():
        while len(seq) < total_steps:
            x_input = np.array(seq[-SEQ_LEN:])[None, :, None]
            x_tensor = torch.from_numpy(x_input).float().to(device)
            y_next = model(x_tensor).cpu().numpy()[0, 0]
            seq.append(y_next)

    return np.array(seq)

# ===================== 6. 训练模型 =====================
# 训练集长度
train_len = int(TOTAL_STEPS * TRAIN_FRAC_LAMBDA / TOTAL_PERIODS)
train_len = max(train_len, SEQ_LEN + 1)

y_train = y_all[:train_len]

X_train, Y_train = create_dataset(y_train, SEQ_LEN)
X_train = torch.from_numpy(X_train).float().to(device)
Y_train = torch.from_numpy(Y_train).float().to(device)

model = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=1).to(device)  # 模型层数与维度设置，e.g., 设为2层

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

# ===================== 7. 外推预测 =====================
y_init = y_all[:SEQ_LEN]
y_pred = autoregressive_predict(model, y_init, TOTAL_STEPS)

# ===================== 8. 画图 =====================
# -------- 图 1：Loss - Epoch --------
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(f"Training Loss (train length = {TRAIN_FRAC_LAMBDA} λ)")
plt.grid(True)
plt.show()

# -------- 图 2：sin(x) 拟合与外推 --------
plt.figure(figsize=(8, 5))
plt.plot(x_all, y_all, label="True sin(x)")
plt.plot(x_all, y_pred, '--', label="LSTM prediction")

# 标注训练区间
plt.axvspan(0, TRAIN_FRAC_LAMBDA * LAMBDA,
            color='gray', alpha=0.2, label="Training region")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"LSTM Extrapolation of sin(x)\n(train length = {TRAIN_FRAC_LAMBDA} λ)")
plt.legend()
plt.grid(True)
plt.show()
