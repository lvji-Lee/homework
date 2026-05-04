import os
os.system('cls')

import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

file_path = r"E:\学习资源！！！！！！\数字制造技术\task5\sine_noised2.xlsx"
column_name = "Noised data3"

data = pd.read_excel(file_path)
data.columns = data.columns.str.strip()
print("Excel 列名如下：")
print(list(data.columns))

x = data[column_name].astype(float).to_numpy()

# 小波分解（Haar，最多8层）
wavelet = 'haar'
max_level = min(8, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
coeffs = pywt.wavedec(x, wavelet, level=max_level)

print(f"实际分解层数: {max_level}")

# ==============================
# 1 个三联图（原始、近似 s^j、细节 d^j）
# ==============================
x_axis = np.arange(len(x))
for j in range(1, max_level + 1):
    # 以第 j 层为目标重新分解（便于单独拿到 cA_j 与 cD_j）
    coeffs_j = pywt.wavedec(x, wavelet, level=j)   # [cA_j, cD_j, cD_{j-1}, ..., cD_1]

    # 仅保留第 j 层近似函数 -> s^j
    coeffs_sj = [coeffs_j[0]] + [np.zeros_like(c) for c in coeffs_j[1:]]
    s_j = pywt.waverec(coeffs_sj, wavelet)[:len(x)]

    # 仅保留第 j 层细节函数 -> d^j（在列表的索引 1 处）
    coeffs_dj = [np.zeros_like(coeffs_j[0])] + [np.zeros_like(c) for c in coeffs_j[1:]]
    coeffs_dj[1] = coeffs_j[1]
    d_j = pywt.waverec(coeffs_dj, wavelet)[:len(x)]

    # 获取上一层的近似图像 s^{j-1}
    if j > 1:
        coeffs_prev = pywt.wavedec(x, wavelet, level=j-1)  # [cA_{j-1}, cD_{j-1}, ..., cD_1]
        coeffs_sj_minus1 = [coeffs_prev[0]] + [np.zeros_like(c) for c in coeffs_prev[1:]]
        s_j_minus1 = pywt.waverec(coeffs_sj_minus1, wavelet)[:len(x)]
    else:
        s_j_minus1 = x  # 当 j=1 时，上一层的近似就是原始信号

    plt.close('all')
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2], hspace=0.35, wspace=0.3)

    ax_top = fig.add_subplot(gs[0, :])   
    ax_l   = fig.add_subplot(gs[1, 0])   # 左下 s^j
    ax_r   = fig.add_subplot(gs[1, 1])   # 右下 d^j

    fig.suptitle(f"Haar WT (a = 2^{j})", fontsize=14, fontweight='bold', y=0.98)

    ax_top.plot(x_axis, s_j_minus1, linewidth=1.5)
    ax_top.set_title(r'$s^{%d}$' % (j-1) if j > 1 else '$z \; (= s^{0})$', pad=6)
    ax_top.set_xlim(0, len(x_axis)-1)
    ax_top.grid(True, linestyle='--', alpha=0.4)
    ax_top.set_ylabel('Amplitude')

    x_sj = np.linspace(0, len(x)-1, len(s_j)) / (2**j)
    x_dj = np.linspace(0, len(x)-1, len(d_j)) / (2**j)

    # 左下：当前层的近似 s^j
    ax_l.plot(x_sj, s_j, linewidth=1.2)
    ax_l.set_title(rf'$s^{j}$'.replace('j', str(j)))
    ax_l.set_xlabel('x (scaled)')
    ax_l.set_ylabel(rf'$s^{j}$'.replace('j', str(j)))
    ax_l.grid(True, linestyle='--', alpha=0.4)

    # 右下：当前层的细节 d^j
    ax_r.plot(x_dj, d_j, linewidth=1.2)
    ax_r.axhline(0, linewidth=0.8, linestyle=':')
    ax_r.set_title(rf'$d^{j}$'.replace('j', str(j)))
    ax_r.set_xlabel('x (scaled)')
    ax_r.set_ylabel(rf'$d^{j}$'.replace('j', str(j)))
    ax_r.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==============================
# 反变换（重构信号）并与原始信号比较
# ==============================
x_reconstructed = pywt.waverec(coeffs, wavelet)[:len(x)]

# 计算误差
mse = np.mean((x - x_reconstructed) ** 2)
print(f"\n重构完成 ✅  均方误差 MSE = {mse:.6e}")

# ==============================
# 绘制原始信号 vs 重构信号
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(x_axis, x, label='Original Signal', linewidth=1.5)
plt.plot(x_axis, x_reconstructed, '--', label='Reconstructed Signal', linewidth=1.2)
plt.title(f"Haar Wavelet Reconstruction ", fontsize=13)
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ==============================
# 绘制误差曲线
# ==============================
error = x - x_reconstructed

plt.figure(figsize=(10, 3.5))
plt.plot(x_axis, error, color='red', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Reconstruction Error (x - x_reconstructed)", fontsize=13)
plt.xlabel("Sample Index")
plt.ylabel("Error Amplitude")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f"误差统计：")
print(f"  均方误差 (MSE) = {mse:.6e}")
print(f"  最大绝对误差 = {np.max(np.abs(error)):.6e}")
print(f"  平均绝对误差 = {np.mean(np.abs(error)):.6e}")
