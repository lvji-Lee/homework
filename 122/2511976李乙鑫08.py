"""
Hilber-Huang Transform (HHT) 实现示例
-----------------------------------
包含:
1. EMD: 经验模态分解
2. HSA: 希尔伯特谱分析 (Hilbert Spectrum)

作业任务: 修改！！
  - 在开头定义目标函数 x(t)
  - 自动分解为若干 IMF
  - 对每个 IMF 做 Hilbert 变换
  - 绘制 IMF 分量图、Hilbert 3D 谱图、Hilbert 热力图
"""

import os
os.system('cls')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ======================== 1️⃣ 定义信号 ==========================
fs = 1000          # 采样频率
T = 1.0            # 信号时长（秒）
t = np.linspace(0, T, int(fs * T), endpoint=False)

# >>>>>> 在这里修改目标函数 x(t) <<<<<<
# 例子：一个 5 Hz 正弦 + 一个调频分量 (10+20t)Hz + 线性趋势
x = (
    6 * np.sin(2 * np.pi * 5 * t + np.pi / 4)
    + 63 * np.sin(2 * np.pi * (10 + 20 * t) * t + np.pi / 6)
    + 0.5 * t
)
# ===============================================================


# ======================== 2️⃣ 辅助函数 ==========================
def local_extrema(x):
    """
    返回 (极大索引, 极小索引)，包含端点保护。
    端点按一阶差分的符号决定是否并入极值，以减少 EMD 过早停止。
    """
    x = np.asarray(x)
    dx = np.diff(x)
    if dx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    s = np.sign(dx)
    s[s == 0] = 1
    ds = np.diff(s)

    max_idx = np.where(ds < 0)[0] + 1
    min_idx = np.where(ds > 0)[0] + 1

    # 端点处理：把可能的端点极值并入
    if dx[0] < 0:      # 一开始在下降，0 可能是极大
        max_idx = np.r_[0, max_idx]
    elif dx[0] > 0:    # 一开始在上升，0 可能是极小
        min_idx = np.r_[0, min_idx]

    if dx[-1] > 0:     # 末端在上升，最后一点可能是极大
        max_idx = np.r_[max_idx, len(x) - 1]
    elif dx[-1] < 0:   # 末端在下降，最后一点可能是极小
        min_idx = np.r_[min_idx, len(x) - 1]

    # 去重并排序
    max_idx = np.unique(max_idx)
    min_idx = np.unique(min_idx)
    return max_idx, min_idx


def sift_once(t, x):
    """
    一次 sifting：
      - 用三次样条拟合上下包络
      - 得到包络均值 m 和细化信号 h = x - m
    通过端点兜底保证样条尽量可用。
    """
    t = np.asarray(t)
    x = np.asarray(x)

    max_idx, min_idx = local_extrema(x)

    # 兜底：如果某类极值太少，把端点强行并进去
    if len(max_idx) < 2:
        max_idx = np.unique(np.r_[0, max_idx, len(x) - 1])
    if len(min_idx) < 2:
        min_idx = np.unique(np.r_[0, min_idx, len(x) - 1])

    try:
        upper = CubicSpline(t[max_idx], x[max_idx], bc_type="natural")(t)
        lower = CubicSpline(t[min_idx], x[min_idx], bc_type="natural")(t)
    except Exception:
        # 若样条失败，返回失败信号
        return x, np.zeros_like(x), False

    m = 0.5 * (upper + lower)
    h = x - m
    return h, m, True


def emd(t, x, sd_thresh=0.08, max_imfs=16, s_number=4, max_sift=600):
    """
    改进版 EMD：
    - 使用 S-number 准则：需要连续 s_number 次满足 IMF 条件才停止当前 IMF 的 sifting。
    - 提高 max_imfs / max_sift，使 IMF 数量更多、分解更细。
    """
    t = np.asarray(t)
    x = np.asarray(x, dtype=float)

    r = x.copy()   # 残差
    imfs = []

    for _ in range(max_imfs):
        h = r.copy()
        s_ok = 0   # 连续满足 IMF 条件的次数

        for _ in range(max_sift):
            h1, m, ok = sift_once(t, h)
            if not ok:
                # 无法继续 sifting，认为当前残差已无有效模式
                return imfs, r

            # 标准差准则（形状变化程度）
            sd = np.sum((h - h1) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h1

            # IMF 判据：
            # 1) 零交叉数与极值数之差不超过 1
            # 2) 包络均值足够小，或者 sd 足够小
            max_idx, min_idx = local_extrema(h)
            zero_cross = np.where(np.diff(np.sign(h)) != 0)[0]
            imf_like = abs((len(max_idx) + len(min_idx)) - len(zero_cross)) <= 1

            mean_env_small = (np.mean(np.abs(m)) /
                              (np.max(np.abs(h)) + 1e-12)) < 0.05

            if (imf_like and mean_env_small) or (sd < sd_thresh):
                s_ok += 1
            else:
                s_ok = 0

            if s_ok >= s_number:
                break

        imf = h.copy()
        imfs.append(imf)
        r = r - imf

        # 残差停止条件：极值总数 < 2，或能量非常小
        ext_total = len(local_extrema(r)[0]) + len(local_extrema(r)[1])
        if ext_total < 2 or (np.linalg.norm(r) < 1e-6 * np.linalg.norm(x)):
            break

    return imfs, r


def hilbert_spectrum(t, imfs):
    """
    Hilbert 谱分析：
      对每个 IMF 做 Hilbert 变换，得到瞬时幅值 A 和瞬时频率 f。
    """
    dt = t[1] - t[0]
    A_list, F_list = [], []
    for c in imfs:
        z = hilbert(c)
        A = np.abs(z)
        phi = np.unwrap(np.angle(z))
        f = np.gradient(phi, dt) / (2 * np.pi)
        A_list.append(A)
        F_list.append(f)
    return A_list, F_list


def build_HSA_map(t, F_list, A_list, fmax=60, nf=200):
    """
    构建时间-频率-幅值矩阵 H (nf × T)：
      行：频率；列：时间；元素：该时刻附近所有 IMF 在该频率上的幅值和。
    """
    Tn = len(t)
    H = np.zeros((nf, Tn))
    f_axis = np.linspace(0, fmax, nf)

    for A, F in zip(A_list, F_list):
        # 频率限制在 [0, fmax]
        F = np.clip(F, 0, fmax)
        idx = np.floor(F / fmax * (nf - 1)).astype(int)
        idx = np.clip(idx, 0, nf - 1)
        for i in range(Tn):
            H[idx[i], i] += A[i]

    return f_axis, H


# ======================== 3️⃣ 执行 EMD ==========================
# 可选：先去趋势，再做 EMD，会更容易分出高频 IMF
# trend = np.poly1d(np.polyfit(t, x, 1))(t)
# x_emd = x - trend
# imfs, r = emd(t, x_emd, sd_thresh=0.08, max_imfs=16, s_number=4, max_sift=600)
# r = r + trend  # 把趋势加回到残差中

# 直接对原信号做 EMD：
# 先用“宽松参数”尽量多分
imfs_all, r_all = emd(t, x, sd_thresh=0.05, max_imfs=20, s_number=5, max_sift=800)
K = 4  # 你想要的层数
imfs = imfs_all[:K]
r = x - np.sum(imfs, axis=0)  # 用前 K 层重算残差


# 绘制 IMF 分量
plt.figure(figsize=(10, 2 * (len(imfs) + 2)))

plt.subplot(len(imfs) + 2, 1, 1)
plt.plot(t, x)
plt.title("Original signal x(t)")

for i, imf in enumerate(imfs):
    plt.subplot(len(imfs) + 2, 1, i + 2)
    plt.plot(t, imf)
    plt.ylabel(f"IMF{i + 1}")
    plt.grid(True, linestyle='--', alpha=0.3)
    if i == 0:
        plt.title("IMFs")

plt.subplot(len(imfs) + 2, 1, len(imfs) + 2)
plt.plot(t, r)
plt.title("Residual r(t)")
plt.xlabel("Time (s)")
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# ======================== 4️⃣ Hilbert 变换 ==========================
A_list, F_list = hilbert_spectrum(t, imfs)
f_axis, H = build_HSA_map(t, F_list, A_list, fmax=60, nf=200)


# ======================== 5️⃣ 绘制 Hilbert 3D 谱图 ==========================
Tg, Fg = np.meshgrid(t, f_axis)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 为了视觉效果，把极端小/大值裁剪一下
vmin = np.min(H)
vmax = np.percentile(H, 99.5)
H_plot = np.clip(H, vmin, vmax)

surf = ax.plot_surface(Tg, Fg, H_plot,
                       cmap=cm.viridis,
                       linewidth=0,
                       antialiased=True)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_zlabel("Amplitude")
ax.set_title("Hilbert Spectrum 3D")

# 自动设置 z 轴范围，避免 zlim 相同的警告
zmin = np.min(H_plot)
zmax = np.max(H_plot)
if np.isclose(zmin, zmax):
    ax.set_zlim(zmin - 1e-9, zmax + 1e-9)
else:
    ax.set_zlim(zmin, zmax)

fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="Amplitude")
plt.tight_layout()
plt.show()


# ======================== 6️⃣ Hilbert 热力图 ==========================
plt.figure(figsize=(9, 4.5))

vmin = np.min(H)
vmax = np.percentile(H, 99.5)  # 抑制极端大值
plt.pcolormesh(t, f_axis, H, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Hilbert Spectrum (Time–Frequency–Amplitude)")
plt.colorbar(label="Amplitude")

plt.tight_layout()
plt.show()
