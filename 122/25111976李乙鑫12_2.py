import os
os.system('cls')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 设置随机种子
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)

# ================================================================
# 1. 数据生成
# ================================================================
N = 300
mu, sigma = 0.0, 0.2

x1 = np.linspace(-2, 2, N)
y1_noise = 3*x1 + 6 + np.random.normal(mu, sigma, size=x1.shape)

x2 = np.linspace(-2, 2, N)
y2_noise = x2**2 + np.random.normal(mu, sigma, size=x2.shape)

x3 = np.linspace(0, 2*np.pi, N)
y3_noise = np.sin(x3) + np.random.normal(mu, sigma, size=x3.shape)

# ================================================================
# 2. 激活函数
# ================================================================
def get_activation(name):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name == "leakyrelu": return nn.LeakyReLU(0.01)
    raise ValueError(f"Unknown activation: {name}")

# ================================================================
# 3. 权重初始化
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
            raise ValueError(f"Unknown init type: {init_type}")
        nn.init.zeros_(m.bias)

# ================================================================
# 4. 构建模型
# ================================================================
def build_mlp(activation="tanh", init_type="xavier"):
    act = get_activation(activation)
    model = nn.Sequential(
        nn.Linear(1, 100), act,
        nn.Linear(100, 100), act,
        nn.Linear(100, 100), act,
        nn.Linear(100, 100), act,
        nn.Linear(100, 100), act,
        nn.Linear(100, 1)
    )
    model.apply(lambda m: init_weights(m, init_type))
    return model

# ================================================================
# 5. 单次训练
# ================================================================
def train_one_function(x, y, config):
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = build_mlp(config["activation"], config["init"])

    if config["optimizer"] == "adam":
        opt = optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        opt = optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

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
# 6. 多配置对比绘图（自动生成图例）
# ================================================================
def run_experiments_and_plot_multi(configs):

    results_linear = []
    results_quad = []
    results_sine = []
    legend_texts = []

    for cfg in configs:
        # 自动生成 legend 文本
        legend = f"{cfg['optimizer'].upper()} | {cfg['activation']} | {cfg['init']}"
        print(f"Training: {legend}")

        l1 = train_one_function(x1, y1_noise, cfg)
        l2 = train_one_function(x2, y2_noise, cfg)
        l3 = train_one_function(x3, y3_noise, cfg)

        results_linear.append(l1)
        results_quad.append(l2)
        results_sine.append(l3)
        legend_texts.append(legend)

    plt.figure(figsize=(15, 4))

    # (1) Linear
    plt.subplot(1, 3, 1)
    for loss, lab in zip(results_linear, legend_texts):
        plt.plot(loss, label=lab)
    plt.title("J-Epoch (y = 3x + 6)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss J")
    plt.legend()

    # (2) Quadratic
    plt.subplot(1, 3, 2)
    for loss, lab in zip(results_quad, legend_texts):
        plt.plot(loss, label=lab)
    plt.title("J-Epoch (y = x²)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss J")
    plt.legend()

    # (3) Sine
    plt.subplot(1, 3, 3)
    for loss, lab in zip(results_sine, legend_texts):
        plt.plot(loss, label=lab)
    plt.title("J-Epoch (y = sin x)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss J")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ================================================================
# 7. 只需要在这里批量配置对比实验
# ================================================================
configs = [
    {
        "optimizer": "adam",
        "activation": "tanh",
        "init": "xavier",
        "batch_size": 32,
        "lr": 0.003,
        "epochs": 500
    },
    {
        "optimizer": "adam",
        "activation": "tanh",
        "init": "normal",
        "batch_size": 32,
        "lr": 0.01,
        "epochs": 500
    },
    {
        "optimizer": "adam",
        "activation": "tanh",
        "init": "he",
        "batch_size": 32,
        "lr": 0.003,
        "epochs": 500
    }
]

# 运行：一次性画多条曲线
run_experiments_and_plot_multi(configs)
