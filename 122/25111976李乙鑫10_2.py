# -*- coding: utf-8 -*-
"""
SiC.csv 峰形拟合（Peak fitting）
模型： y(x) = (b0 + b1*x) + A * [ eta * L(x; x0, w) + (1-eta) * G(x; x0, w) ]
其中：
  - 线性基线：b0 + b1*x
  - 伪Voigt峰：eta∈[0,1]，w为FWHM，A为峰高（相对基线）
  - G 使用 FWHM 形式的高斯:  G = exp( -4*ln2 * ((x-x0)^2 / w^2) )
  - L 使用 FWHM 形式的洛伦兹: L = 1 / ( 1 + 4 * ((x-x0)^2 / w^2) )

输出：
  - Base line（b0 + b1*x 在 x0 处的值）
  - Peak position（x0）
  - Peak height（A）
  - FWHM（w）
并绘制原始数据与拟合曲线（含注释标注）。
新增功能：分别生成原始图像和拟合图像的独立曲线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ========================= 用户输入：CSV 路径 =========================
CSV_PATH = r"E:\学习资源！！！！！！\数字制造技术\task10\SiC.csv"   # <-- 把这行改成你的文件地址

# ========================= 模型定义 =========================
LN2 = np.log(2.0)

def gaussian_fwhm(x, x0, w):
    """FWHM 参数化的高斯核，峰顶=1"""
    return np.exp(-4.0 * LN2 * ((x - x0) ** 2) / (w ** 2))

def lorentzian_fwhm(x, x0, w):
    """FWHM 参数化的洛伦兹核，峰顶=1"""
    return 1.0 / (1.0 + 4.0 * ((x - x0) ** 2) / (w ** 2))

def pseudo_voigt(x, A, x0, w, eta, b0, b1):
    """
    伪Voigt + 线性基线（总模型）
    A   : 峰高度（相对基线，亦即在 x0 处的增量）
    x0  : 峰位置
    w   : FWHM
    eta : 形状混合系数 ∈ [0,1]；0=纯高斯，1=纯洛伦兹
    b0  : 基线截距
    b1  : 基线斜率
    """
    G = gaussian_fwhm(x, x0, w)
    L = lorentzian_fwhm(x, x0, w)
    return (b0 + b1 * x) + A * (eta * L + (1.0 - eta) * G)

# ========================= 初值与约束 =========================
def rough_initial_guess(x, y):
    """根据数据粗略估计初值 (A, x0, w, eta, b0, b1)"""
    # 峰位置初猜：最大值位置
    idx_peak = np.argmax(y)
    x0_init = float(x[idx_peak])

    # 基线初猜：取两端各10%数据做线性回归
    n = len(x)
    k = max(1, int(0.1 * n))
    edge_mask = np.r_[np.arange(k), np.arange(n - k, n)]
    X = np.column_stack([np.ones(edge_mask.size), x[edge_mask]])
    coef, *_ = np.linalg.lstsq(X, y[edge_mask], rcond=None)
    b0_init, b1_init = coef.tolist()

    # 峰高初猜：峰顶减去在峰位置的线性基线
    A_init = float(max(1e-6, y[idx_peak] - (b0_init + b1_init * x0_init)))

    # 宽度初猜：用半高宽的粗略估计（若失败就用 1/10跨度）
    half = (b0_init + b1_init * x0_init) + 0.5 * A_init
    # 找到与半高最接近的左右点
    left = np.where(y[:idx_peak] <= half)[0]
    right = idx_peak + np.where(y[idx_peak:] <= half)[0]
    if left.size > 0 and right.size > 0:
        w_init = float(x[right[0]] - x[left[-1]])
        if w_init <= 0:
            w_init = float((x.max() - x.min()) / 10.0)
    else:
        w_init = float((x.max() - x.min()) / 10.0)

    # 形状混合：给个中间值
    eta_init = 0.5

    return A_init, x0_init, w_init, eta_init, b0_init, b1_init

def fit_peak(x, y):
    """拟合伪Voigt+线性基线，返回参数与协方差"""
    p0 = rough_initial_guess(x, y)

    # 参数边界
    A_lo = 0.0
    A_hi = np.inf
    x0_lo = float(np.min(x))
    x0_hi = float(np.max(x))
    w_lo = (x0_hi - x0_lo) / 1e4  # 宽度不能为0，给很小的正数
    w_hi = (x0_hi - x0_lo) * 2.0
    eta_lo, eta_hi = 0.0, 1.0
    b0_lo, b0_hi = -np.inf, np.inf
    b1_lo, b1_hi = -np.inf, np.inf

    bounds = (
        [A_lo, x0_lo, w_lo, eta_lo, b0_lo, b1_lo],
        [A_hi, x0_hi, w_hi, eta_hi, b0_hi, b1_hi],
    )

    popt, pcov = curve_fit(
        pseudo_voigt, x, y, p0=p0, bounds=bounds, maxfev=20000
    )
    return popt, pcov

# ========================= 新增绘图函数 =========================
def plot_individual_curves(x, y, y_fit, popt, baseline_at_x0, peak_position, peak_height, FWHM):
    """分别绘制原始数据和拟合曲线的独立图像"""
    A, x0, w, eta, b0, b1 = popt
    baseline = b0 + b1 * x
    half_level = baseline_at_x0 + 0.5 * peak_height
    
    # 1. 原始数据独立图像
    plt.figure(figsize=(8, 4.8), dpi=120)
    plt.plot(x, y, 'b-', lw=1.5, label="Raw data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original Data - SiC.csv")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. 拟合曲线独立图像（包含基线）
    plt.figure(figsize=(8, 4.8), dpi=120)
    plt.plot(x, y_fit, 'r-', lw=1.5, label="Fitted curve")
    plt.plot(x, baseline, 'g--', lw=1.0, label="Baseline (linear)")
    
    # 标注峰位与半高宽
    plt.axvline(x0, color="k", lw=0.8, linestyle=":", alpha=0.7)
    plt.axhline(half_level, color="k", lw=0.8, linestyle=":", alpha=0.7)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fitted Curve - Pseudo-Voigt + Linear Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========================= 主流程 =========================
def main():
    # 1) 读取数据
    df = pd.read_csv(CSV_PATH, header=None)
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：第一列 x，第二列 y")
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    # 2) 拟合
    popt, pcov = fit_peak(x, y)
    A, x0, w, eta, b0, b1 = popt
    y_fit = pseudo_voigt(x, *popt)

    # 3) 计算四个参数
    baseline_at_x0 = b0 + b1 * x0      # 基线（在峰位 x0 处）
    peak_position = x0                 # 峰位置
    peak_height = A                    # 峰高度（相对基线）
    FWHM = w                           # 半高宽

    # 4) 打印结果
    print("===== Peak fitting results (pseudo-Voigt + linear baseline) =====")
    print(f"Base line @ x0      : {baseline_at_x0:.6g}  (b0={b0:.6g}, b1={b1:.6g})")
    print(f"Peak position (x0)  : {peak_position:.6g}")
    print(f"Peak height (A)     : {peak_height:.6g}")
    print(f"FWHM (w)            : {FWHM:.6g}")
    print(f"Shape mix (eta)     : {eta:.6g}   (0=Gaussian, 1=Lorentzian)")

    # 5) 绘制对比图（保持原有功能）
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    ax.plot(x, y, 'bo-', lw=1.2, markersize=3, label="Raw data")
    ax.plot(x, y_fit, 'r-', lw=1.5, label="Fitted curve")

    # 基线（以拟合参数绘制）
    baseline = b0 + b1 * x
    ax.plot(x, baseline, 'g--', lw=1.0, label="Baseline (linear)")

    # 标注峰位与半高宽
    ax.axvline(x0, color="k", lw=0.8, linestyle=":")
    ax.text(x0, baseline_at_x0 + A, f"x0={x0:.4g}", ha="center", va="bottom")

    # 半高水平线
    half_level = baseline_at_x0 + 0.5 * A
    ax.axhline(half_level, color="k", lw=0.8, linestyle=":")
    ax.text(x.min(), half_level, "half max", va="bottom")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Peak fitting of SiC.csv (pseudo-Voigt + linear baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6) 新增：分别绘制原始数据和拟合曲线的独立图像
    plot_individual_curves(x, y, y_fit, popt, baseline_at_x0, peak_position, peak_height, FWHM)

if __name__ == "__main__":
    main()