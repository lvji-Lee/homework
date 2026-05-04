import os
os.system('cls')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 设置随机种子，保证可重复
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)


# ================================================================
# 1. 数据生成（与你原来一致）
# ================================================================
N = 300
mu, sigma = 0.0, 0.2

# Linear: y = 3x + 6
x1 = np.linspace(-2, 2, N)
y1_clean = 3 * x1 + 6
y1_noise = y1_clean + np.random.normal(mu, sigma, size=x1.shape)

# Quadratic: y = x^2
x2 = np.linspace(-2, 2, N)
y2_clean = x2**2
y2_noise = y2_clean + np.random.normal(mu, sigma, size=x2.shape)

# Sine: y = sin(x)
x3 = np.linspace(0, 2*np.pi, N)
y3_clean = np.sin(x3)
y3_noise = y3_clean + np.random.normal(mu, sigma, size=x3.shape)


# ================================================================
# 2. Activation function factory
# ================================================================
def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01)
    else:
        raise ValueError("Unknown activation function")


# ================================================================
# 3. Weight initialization
# ================================================================
def init_weights(m, init_type):
    if isinstance(m, nn.Linear):
        if init_type == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif init_type == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        else:
            raise ValueError("Unknown init type")
        nn.init.zeros_(m.bias)


# ================================================================
# 4. Build MLP model
# ================================================================
def build_mlp(activation="tanh", init_type="xavier"):
    act = get_activation(activation)

    model = nn.Sequential(
        nn.Linear(1, 100), act,
        nn.Linear(100, 100), act,
        nn.Linear(100, 100), act,
        nn.Linear(100, 100), act,    # 第4层（新增加这一行）
        nn.Linear(100, 100), act,    # 第5层（新增加这一行）
        nn.Linear(100, 1)
    )

    model.apply(lambda m: init_weights(m, init_type))
    return model


# ================================================================
# 5. Train once for one function, return J-Epoch
# ================================================================
def train_one_function(x, y, config):
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = build_mlp(config["activation"], config["init"])

    # choose optimizer
    if config["optimizer"] == "adam":
        opt = optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        opt = optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=config["lr"])
    else:
        raise ValueError("Unknown optimizer")

    loss_fn = nn.MSELoss()

    losses = []

    for epoch in range(config["epochs"]):
        for bx, by in loader:
            pred = model(bx)
            loss = loss_fn(pred, by)

            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(loss.item())

    return losses


# ================================================================
# 6. Train all three functions and plot J–Epoch curves
# ================================================================
def run_experiments_and_plot(config):

    loss1 = train_one_function(x1, y1_noise, config)
    loss2 = train_one_function(x2, y2_noise, config)
    loss3 = train_one_function(x3, y3_noise, config)

    plt.figure(figsize=(15, 4))

    # (1) Linear
    plt.subplot(1, 3, 1)
    plt.plot(loss1, 'r')
    plt.title("J-Epoch (y = 3x + 6)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss J")

    # (2) Quadratic
    plt.subplot(1, 3, 2)
    plt.plot(loss2, 'g')
    plt.title("J-Epoch (y = x²)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss J")

    # (3) Sine
    plt.subplot(1, 3, 3)
    plt.plot(loss3, 'b')
    plt.title("J-Epoch (y = sin x)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss J")

    plt.tight_layout()
    plt.show()


# ================================================================
# 7. 训练器参数修改位置
# ================================================================
config = {
    "optimizer": "adam",        # 可改："adam" / "sgd" / "rmsprop"
    "activation": "tanh",       # 可改："relu" / "tanh" / "sigmoid" / "leakyrelu"
    "init": "xavier",           # 可改："xavier" / "he" / "normal"
    "batch_size": 1,
    "lr": 0.003,
    "epochs": 500
}

# 运行整个流程
run_experiments_and_plot(config)
