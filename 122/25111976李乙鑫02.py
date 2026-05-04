import os
os.system('cls')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 Excel 文件
file_path = r"E:\Python_code package\pro1\homework\load1.0N FrictionForce.xlsx"
df = pd.read_excel(file_path)

time = df.iloc[:,0]   # 第一列是时间
friction = df.iloc[:,1]  # 第二列是摩擦力

# ==========================
# 简单移动平均 (SMA) 函数
# ==========================
def sma_filter(signal, M):
    if M % 2 == 0:
        raise ValueError("窗口大小 M 必须是奇数，例如 3, 5, 7 ...")
    L = M // 2
    padded = np.pad(signal, (L, L), mode='edge')
    kernel = np.ones(M) / M
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

# ==========================
# 高斯加权移动平均 (Gaussian WMA) 函数
# ==========================
def gaussian_wma_filter(signal, M, sigma=None):
    if M % 2 == 0:
        raise ValueError("窗口大小 M 必须是奇数，例如 3, 5, 7 ...")
    L = M // 2
    if sigma is None:
        sigma = M / 6   # 经验值：约 3σ ≈ 窗口一半
    # 构造高斯权重
    x = np.arange(-L, L+1)
    weights = np.exp(-0.5 * (x / sigma)**2)
    weights /= np.sum(weights)  # 归一化
    # padding
    padded = np.pad(signal, (L, L), mode='edge')
    smoothed = np.convolve(padded, weights, mode='valid')
    return smoothed

# ==========================
# 设置滤波窗口大小 (M)
# ==========================
M =401 # 👈 在这里调整 M（必须是奇数）

friction_sma = sma_filter(friction.values, M)
friction_gwma = gaussian_wma_filter(friction.values, M)

# ==========================
# 设置目标文件夹
# ==========================
output_dir = r"E:\学习资源！！！！！！\数字制造技术\task2\photo"
os.makedirs(output_dir, exist_ok=True)  # 如果不存在则自动创建

# ==========================
# 画图并保存 (文件名带 M)
# ==========================

# 对比图
plt.figure(figsize=(10,6))
plt.plot(time, friction, label="Raw Friction Force", linewidth=1, alpha=0.6)
plt.plot(time, friction_sma, label=f"SMA (M={M})", linewidth=2, color='red')
plt.plot(time, friction_gwma, label=f"Gauss WMA (M={M})", linewidth=2, color='black')
plt.title(f"Friction Force with SMA & Gaussian WMA Filtering (M={M})")
plt.xlabel("Time [ms]")
plt.ylabel("Friction Force [N]")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"friction_comparison_M{M}.svg"), format="svg")
plt.close()

# SMA 图
plt.figure(figsize=(10,6))
plt.plot(time, friction, label="Raw Friction Force", linewidth=1, alpha=0.6)
plt.plot(time, friction_sma, label=f"SMA (M={M})", linewidth=2, color='red')
plt.title(f"Friction Force with SMA Filtering (M={M})")
plt.xlabel("Time [ms]")
plt.ylabel("Friction Force [N]")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"friction_sma_M{M}.svg"), format="svg")
plt.close()

# Gaussian WMA 图
plt.figure(figsize=(10,6))
plt.plot(time, friction, label="Raw Friction Force", linewidth=1, alpha=0.6)
plt.plot(time, friction_gwma, label=f"Gauss WMA (M={M})", linewidth=2, color='black')
plt.title(f"Friction Force with Gaussian WMA Filtering (M={M})")
plt.xlabel("Time [ms]")
plt.ylabel("Friction Force [N]")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"friction_gauss_wma_M{M}.svg"), format="svg")
plt.close()
