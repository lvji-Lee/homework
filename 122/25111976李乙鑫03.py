import os
os.system('cls')

import numpy as np
import matplotlib.pyplot as plt

# z(t) = A * sin(2π f t)
def z_of_f(f, A=1.0, fmax=300.0, cycles=2):
    if f <= 0:
        f = 1e-6
    dt = 1.0 / (8* fmax)         # sampling rate = 10 * fmax
    t_end = cycles / f               # two cycles
    t = np.arange(0.0, t_end, dt)
    z = A * np.sin(2 * np.pi * f * t)
    return t, z

# ==========================
# Filters (严格按课件：mode='same'，不做padding)
# ==========================
def sma_filter(signal, M):
    if M % 2 == 0:
        raise ValueError("窗口大小 M 必须是奇数，例如 3, 5, 7 ...")
    kernel = np.ones(M) / M
    return np.convolve(signal, kernel, mode='valid')

def gaussian_wma_filter(signal, M, sigma=None):
    if M % 2 == 0:
        raise ValueError("窗口大小 M 必须是奇数，例如 3, 5, 7 ...")
    L = M // 2
    if sigma is None:
        sigma = M / 4   # 经验值：约 3σ ≈ 窗口一半
    x = np.arange(-L, L+1)
    weights = np.exp(-0.5 * (x / sigma)**2)
    weights /= np.sum(weights)  # 归一化
    return np.convolve(signal, weights, mode='same')

# ==========================
# amplitude (peak-to-peak / 2)
# ==========================
def amp(x):
    return (np.max(x) - np.min(x)) / 2.0

# ==========================
# compute η curve by time-domain convolution and P2P amplitude
# ==========================
def eta_curve_SMA(f_grid, M, fmax, cycles=2):
    eta = []
    for f in f_grid:
        _, z = z_of_f(f, A=1.0, fmax=fmax, cycles=cycles)
        u = sma_filter(z, M)
        val = 0.0 if amp(z) == 0 else amp(u) / amp(z)
        eta.append(val)
    return np.array(eta)

def eta_curve_GWMA(f_grid, M, sigma, fmax, cycles=2):
    eta = []
    for f in f_grid:
        _, z = z_of_f(f, A=1.0, fmax=fmax, cycles=cycles)
        u = gaussian_wma_filter(z, M, sigma)
        val = 0.0 if amp(z) == 0 else amp(u) / amp(z)
        eta.append(val)
    return np.array(eta)

# ==========================
# cutoff finder (η ≈ 0.7079)
# ==========================
def cutoff_freq(f_grid, eta, thr=0.7079):
    for k in range(1, len(eta)):
        if eta[k] < thr <= eta[k-1]:
            x0, x1 = f_grid[k-1], f_grid[k]
            y0, y1 = eta[k-1], eta[k]
            return x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
    return f_grid[-1]

# ==========================
# analytical formula (你原来就有，保留)
# ==========================
def eta_analytic(f, sigma, i=1.0):
    return (1/np.pi) * np.exp(-2*(np.pi**2)*(f**2)*(sigma**2)) * np.abs(np.sin(2*np.pi*f*i))

# ==========================
# Parameters
# ==========================
fmax = 300.0
f_grid = np.arange(0.1, fmax + 0.1, 1)   # 0.1 Hz step
M_list = np.array([5,7,9,11,13,15,17])
sigma_list = M_list / 4.0
M_wma = 51
cycles = 2        # 与课件一致：2 个周期

# ==========================
# η–f: SMA
# ==========================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for M in M_list:
    eta = eta_curve_SMA(f_grid, M, fmax, cycles=cycles)
    plt.plot(f_grid, eta, label=f"M = {M}")
plt.xlabel("Frequency f (Hz)")
plt.ylabel("Gain η")
plt.title("SMA: η–f ")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.legend()


# ==========================
# η–f: Gaussian WMA（M 与 SMA 一致变化；图例仅显示 M）
# ==========================
plt.subplot(1,2,2)
for M in M_list:
    # 这里让 sigma=None，内部会用默认 sigma = M/4
    eta = eta_curve_GWMA(f_grid, M, None, fmax, cycles=cycles)
    plt.plot(f_grid, eta, label=f"M = {M}")
plt.xlabel("Frequency f (Hz)")
plt.ylabel("Gain η")
plt.title("Gaussian WMA: η–f (σ = M/4)")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================
# Cutoff curves
# ==========================
cut_sma = [cutoff_freq(f_grid, eta_curve_SMA(f_grid, M, fmax, cycles=cycles)) for M in M_list]
# WMA 也随 M_list 变化（sigma 用默认 M/4），图例仅显示 M
cut_wma = [cutoff_freq(f_grid, eta_curve_GWMA(f_grid, M, None, fmax, cycles=cycles)) for M in M_list]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(M_list, cut_sma, "o-")
plt.xlabel("M")
plt.ylabel("Cutoff f_c (Hz)")
plt.title("Cutoff vs M (SMA)")
plt.grid(True)
plt.ylim(bottom=0)

plt.subplot(1,2,2)
plt.plot(M_list, cut_wma, "o-")
plt.xlabel("M")
plt.ylabel("Cutoff f_c (Hz)")
plt.title("Cutoff vs M (WMA, σ = M/4)")
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# ==========================
# Analytical plot（保留）
# ==========================
def eta_analytic(f, sigma, i=1.0, A=1.0):
    return (A/np.pi) * np.exp(-2*(np.pi**2)*(f**2)*(sigma**2)) * np.abs(np.sin(2*np.pi*f*i))

def safe_normalize(y):
    m = np.max(y)
    return y / m if m > 0 else y

# ===== 参数 =====
i = 0.1
A = 1.0
f = np.linspace(0.1, 1.0, 500)

# 不同 M 对应 σ = M/4（按你的设定）
M_list = [ 7, 9, 11, 13, 15, 17]

# 选择归一化方式: "per_curve" 或 "global"
normalize_mode = "per_curve"   # 改成 "global" 看全局归一化效果

# 先算所有原始曲线
curves = []
for M in M_list:
    sigma = M / 4.0
    eta = eta_analytic(f, sigma, i=i, A=A)
    curves.append((M, sigma, eta))

# 归一化
if normalize_mode == "per_curve":
    curves_norm = [(M, sigma, safe_normalize(eta)) for (M, sigma, eta) in curves]
elif normalize_mode == "global":
    global_max = max(np.max(eta) for (_, _, eta) in curves)
    scale = (1.0 / global_max) if global_max > 0 else 1.0
    curves_norm = [(M, sigma, eta * scale) for (M, sigma, eta) in curves]
else:
    raise ValueError("normalize_mode 只能是 'per_curve' 或 'global'")

# 绘图
plt.figure(figsize=(7,5))
for M, sigma, eta_n in curves_norm:
    plt.plot(f, eta_n, label=f"M={M} (σ={sigma:.2f})")

plt.xlabel("Frequency f")
plt.ylabel(r"Normalized $\eta$")

title_suffix = "" if normalize_mode == "per_curve" else "Global"

plt.title(
    r"Normalized $\eta(f) = \frac{A}{\pi} e^{-2\pi^2 f^2\sigma^2}\,|\sin(2\pi f i)|$"
    + f"\n($\sigma=M/4$ )"
)

plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# === SMA 解析解（来自公式）===
def sma_eta(f, M):
    # η(f) = sin(Mπf) / (Mπf)
    x = M * np.pi * f
    return np.where(np.abs(f) < 1e-12, 1.0, np.sin(x) / x)

# 频率范围
f = np.linspace(0, 2.0, 1601)  # 0~2 cycles/sample

# 不同 M
M_list = [5, 7, 9, 11, 13, 15]

plt.figure(figsize=(7,5))
for M in M_list:
    eta = sma_eta(f, M)
    plt.plot(f, eta, label=f"M={M}")

plt.xlabel("Frequency f (cycles/sample)")
plt.ylabel(r"$\eta(f)$")
plt.title(r"SMA Analytical $\eta(f)=\frac{1}{M\pi f}\sin(M\pi f)$")
plt.ylim(-0.2, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()