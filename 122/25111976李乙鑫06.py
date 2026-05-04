import os
os.system('cls')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体（解决中文警告）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============== Lifting 单层变换 ==============
def dwt_lifting(s):
    s = np.asarray(s, dtype=float)
    if len(s) % 2 == 1:
        s = np.append(s, s[-1])
    
    even = s[0::2].copy()
    odd  = s[1::2].copy()
    
    # Predict
    p = np.array([-1/16, 9/16, 9/16, -1/16])
    even_pad = np.pad(even, (1, 2), mode='edge')
    predict = np.array([np.sum(p * even_pad[k:k+4]) for k in range(len(even))])
    d = odd - predict
    
    # Update
    u = np.array([-1/32, 9/32, 9/32, -1/32])
    d_pad = np.pad(d, (1, 2), mode='edge')
    update = np.array([np.sum(u * d_pad[k:k+4]) for k in range(len(even))])
    a = even + update
    
    return a, d

def idwt_lifting(a, d):
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    
    # 确保 a 和 d 长度一致
    if len(d) < len(a):
        d = np.pad(d, (0, len(a) - len(d)), mode='edge')
    elif len(d) > len(a):
        d = d[:len(a)]
    
    # Inverse Update
    u = np.array([-1/32, 9/32, 9/32, -1/32])
    d_pad = np.pad(d, (1, 2), mode='edge')
    update = np.array([np.sum(u * d_pad[k:k+4]) for k in range(len(a))])
    even = a - update
    
    # Inverse Predict
    p = np.array([-1/16, 9/16, 9/16, -1/16])
    even_pad = np.pad(even, (1, 2), mode='edge')
    predict = np.array([np.sum(p * even_pad[k:k+4]) for k in range(len(even))])
    odd = d + predict
    
    # Merge
    s = np.zeros(len(even) + len(odd))
    s[0::2] = even
    s[1::2] = odd
    return s

# ============== 多层分解/重构 ==============
def wavedec_lifting(x, level):
    a = np.asarray(x, dtype=float).copy()
    details = []
    for _ in range(level):
        a, d = dwt_lifting(a)
        details.append(d)
    return a, details  # 返回 (a_L, [d_1, d_2, ..., d_L])

def waverec_lifting(a, details):
    # 从 d_L 到 d_1 倒序重构
    for d in reversed(details):
        a = idwt_lifting(a, d)
    return a

# ============== 读取数据 ==============
file_path = r"E:\学习资源！！！！！！\数字制造技术\task5\sine_noised2.xlsx"
column_name = "Noised data3"

data = pd.read_excel(file_path)
x = data[column_name].astype(float).to_numpy()
N = len(x)
x_axis = np.arange(N)

max_level = min(8, int(np.floor(np.log2(N))))
print(f"实际分解层数: {max_level}")

# ============== 分层展示（二连图）==============
a_tmp = x.copy()
for j in range(1, max_level + 1):
    a_tmp, d_j = dwt_lifting(a_tmp)
    
    # 创建二连图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # 左图：原始信号 vs 近似信号 s^j
    ax1.plot(x_axis, x, label='原始信号', color='blue', lw=1.2, alpha=0.7)
    ax1.plot(np.linspace(0, N-1, len(a_tmp)), a_tmp, 
             label=f'近似信号 $s^{{{j}}}$', color='red', lw=1.5)
    ax1.set_title(f'Level {j}: 原始信号 vs 近似信号')
    ax1.set_xlabel('样本点')
    ax1.set_ylabel('幅值')
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.4)
    
    # 右图：细节信号 d^j
    ax2.plot(np.linspace(0, N-1, len(d_j)), d_j, 
             label=f'细节信号 $d^{{{j}}}$', color='green', lw=1.2)
    ax2.set_title(f'Level {j}: 细节信号')
    ax2.set_xlabel('样本点')
    ax2.set_ylabel('幅值')
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

# ============== 完整分解与重构 ==============
a_final, details_all = wavedec_lifting(x, max_level)
x_rec = waverec_lifting(a_final, details_all)[:N]

err = x - x_rec
mse = float(np.mean(err**2))
print(f"\n重构完成 ✅  MSE = {mse:.6e}")
print(f"最大误差 = {np.max(np.abs(err)):.6e}")

# ============== 绘图：原始 vs 重构 ==============
plt.figure(figsize=(10, 4))
plt.plot(x_axis, x, label='原始信号', lw=1.5)
plt.plot(x_axis, x_rec, '--', label='重构信号', lw=1.2)
plt.legend()
plt.grid(True, ls='--', alpha=0.4)
plt.title("原始信号 vs 重构信号")
plt.tight_layout()
plt.show()

# ============== 绘图：误差 ==============
plt.figure(figsize=(10, 3))
plt.plot(x_axis, err, lw=1, color='red')
plt.axhline(0, ls='--', lw=0.8, color='black')
plt.title("重构误差")
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.show()
