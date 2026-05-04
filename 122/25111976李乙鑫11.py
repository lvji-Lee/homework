import os
os.system('cls')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
np.random.seed(0)

# -----------------------------
# 1. 生成带白噪声的数据
# -----------------------------
N = 300               # 每个函数采样点数
mu, sigma = 0.0, 0.1  # 白噪声的均值和标准差，可自行修改

# y = 3x + 6，x 取 [-2, 2]
x1 = np.linspace(-2, 2, N)
y1_clean = 3 * x1 + 6
y1_noise = y1_clean + np.random.normal(mu, sigma, size=x1.shape)

# y = x^2，x 取 [-2, 2]
x2 = np.linspace(-2, 2, N)
y2_clean = x2**2
y2_noise = y2_clean + np.random.normal(mu, sigma, size=x2.shape)

# y = sin x，x 取 [0, 2π]
x3 = np.linspace(0, 2 * np.pi, N)
y3_clean = np.sin(x3)
y3_noise = y3_clean + np.random.normal(mu, sigma, size=x3.shape)

# -----------------------------
# 2. LSM 拟合
# -----------------------------
def lsm_fit_linear(x, y):
    """拟合 y ≈ a*x + b"""
    A = np.vstack([x, np.ones_like(x)]).T   # 设计矩阵 [x, 1]
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    a=coeff[0]
    b=coeff[1]  
    return a, b

def lsm_fit_quadratic(x, y):
    """拟合 y ≈ a*x^2 + b*x + c"""
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    a=coeff[0]
    b=coeff[1] 
    c=coeff[2]
    return a, b, c

def lsm_fit_sine(x, y):
    """拟合 y ≈ A*sin(x) + B*cos(x) + C(线性形式)"""
    A = np.vstack([np.sin(x), np.cos(x), np.ones_like(x)]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    A_s=coeff[0]
    B_c=coeff[1] 
    C=coeff[2]
    return A_s, B_c, C

# y = 3x + 6 的 LSM
a1, b1 = lsm_fit_linear(x1, y1_noise)
y1_lsm = a1 * x1 + b1

# y = x^2 的 LSM
a2, b2, c2 = lsm_fit_quadratic(x2, y2_noise)
y2_lsm = a2 * x2**2 + b2 * x2 + c2

# y = sin x 的 LSM：A*sinx + B*cosx + C
A3, B3, C3 = lsm_fit_sine(x3, y3_noise)
y3_lsm = A3 * np.sin(x3) + B3 * np.cos(x3) + C3

print("LSM coefficients:")
print(f"  y ≈ {a1:.3f} * x + {b1:.3f}")
print(f"  y ≈ {a2:.3f} * x^2 + {b2:.3f} * x + {c2:.3f}")
print(f"  y ≈ {A3:.3f} * sin(x) + {B3:.3f} * cos(x) + {C3:.3f}")

# -----------------------------
# 3. 使用 MLP 拟合
# -----------------------------
# 这里用 sklearn 的 MLPRegressor，并通过 Pipeline 做标准化
def train_mlp(x, y, n_layers=5, n_neurons=100):
    """
    训练一个可调层数与神经元数量的 MLP 回归器。

    参数：
        n_layers  : 隐藏层层数（默认 3 层）
        n_neurons : 每层神经元数量（默认每层 50 个）
    """
    x = x.reshape(-1, 1)

    # 根据层数创建 tuple，例如 (50, 50, 50)
    hidden_layers = tuple([n_neurons] * n_layers)

    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='tanh',
            solver='adam',
            max_iter=5000,
            random_state=0
        )
    )

    mlp.fit(x, y)
    return mlp


mlp1 = train_mlp(x1, y1_noise)
mlp2 = train_mlp(x2, y2_noise)
mlp3 = train_mlp(x3, y3_noise)

y1_mlp = mlp1.predict(x1.reshape(-1, 1))
y2_mlp = mlp2.predict(x2.reshape(-1, 1))
y3_mlp = mlp3.predict(x3.reshape(-1, 1))

# -----------------------------
# 4. 画图：原始数据 + LSM + MLP
# -----------------------------
plt.figure(figsize=(15, 4))

# (1) y = 3x + 6
plt.subplot(1, 3, 1)
plt.scatter(x1, y1_noise, s=15, label='Noisy data')
# plt.plot(x1, y1_clean, 'k-', marker='o', markersize=7, markevery=10, linewidth=1.5, label='True function')
# plt.plot(x1, y1_lsm, 'r', label='LSM fit')
# plt.plot(x1, y1_mlp, 'g', label='MLP fit')
plt.title('y = 3x + 6')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# (2) y = x^2
plt.subplot(1, 3, 2)
plt.scatter(x2, y2_noise, s=15, label='Noisy data')
# plt.plot(x2, y2_clean, 'k-', marker='o', markersize=7, markevery=10, linewidth=1.5, label='True function')
# plt.plot(x2, y2_lsm, 'r', label='LSM fit')
# plt.plot(x2, y2_mlp, 'g', label='MLP fit')
plt.title('y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# (3) y = sin x
plt.subplot(1, 3, 3)
plt.scatter(x3, y3_noise, s=15, label='Noisy data')
# plt.plot(x3, y3_clean, 'k-', marker='o', markersize=7, markevery=10, linewidth=1.5, label='True function')
# plt.plot(x3, y3_lsm, 'r', label='LSM fit')
# plt.plot(x3, y3_mlp, 'g', label='MLP fit')
plt.title('y = sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
