import os
os.system('cls')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1) Lifting-based DWT/IDWT
# =========================
def dwt_lifting(s, pad_mode='symmetric'):
    s = np.asarray(s, dtype=float)
    if len(s) % 2 == 1:
        s = np.append(s, s[-1])
    even = s[0::2].copy()
    odd  = s[1::2].copy()
    # Predict
    p = np.array([-1/16, 9/16, 9/16, -1/16])
    even_pad = np.pad(even, (1, 2), mode=pad_mode)
    predict = np.array([np.sum(p * even_pad[k:k+4]) for k in range(len(even))])
    d = odd - predict
    # Update
    u = np.array([-1/32, 9/32, 9/32, -1/32])
    d_pad = np.pad(d, (1, 2), mode=pad_mode)
    update = np.array([np.sum(u * d_pad[k:k+4]) for k in range(len(even))])
    a = even + update
    return a, d

def idwt_lifting(a, d, pad_mode='symmetric'):
    a = np.asarray(a, dtype=float)
    d = np.asarray(d, dtype=float)
    if len(d) < len(a):
        d = np.pad(d, (0, len(a)-len(d)), mode='edge')
    elif len(d) > len(a):
        d = d[:len(a)]
    # inverse update
    u = np.array([-1/32, 9/32, 9/32, -1/32])
    d_pad = np.pad(d, (1, 2), mode=pad_mode)
    update = np.array([np.sum(u * d_pad[k:k+4]) for k in range(len(a))])
    even = a - update
    # inverse predict
    p = np.array([-1/16, 9/16, 9/16, -1/16])
    even_pad = np.pad(even, (1, 2), mode=pad_mode)
    predict = np.array([np.sum(p * even_pad[k:k+4]) for k in range(len(even))])
    odd = d + predict
    s = np.zeros(len(even)+len(odd))
    s[0::2], s[1::2] = even, odd
    return s

def wavedec_lifting(x, level, pad_mode='symmetric'):
    """
    逐层分解：details = [d1(最细), d2, ..., dL(最粗)]
    """
    a = np.asarray(x, float).copy()
    details = []
    for _ in range(level):
        a, d = dwt_lifting(a, pad_mode)
        details.append(d)  # d1..dL（由细到粗）
    return a, details

def waverec_lifting(a, details, pad_mode='symmetric'):
    """
    从最粗到最细按逆序回拼：dL..d1
    """
    for d in reversed(details):
        a = idwt_lifting(a, d, pad_mode)
    return a

# =========================
# 2) 可视化
# =========================
def plot_series(reference, curves: dict, title, figsize=(12,4)):
    plt.figure(figsize=figsize)
    plt.plot(reference, label='原始', lw=2.0, alpha=0.6)  # 原始稍粗半透明
    for name, y in curves.items():
        plt.plot(y, label=name, lw=1.2)
    plt.grid(True, ls='--', alpha=0.35)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.show()

def plot_outlier_compare(x_raw, x_clean, mask):
    idx = np.arange(len(x_raw))
    plt.figure(figsize=(12,3.5))
    plt.plot(x_raw, label='去除前', lw=1.4)
    plt.plot(x_clean, label='去除后', lw=1.2)
    if mask.any():
        plt.scatter(idx[mask], x_raw[mask], s=25, color='r', marker='x', label='异常点')
    plt.legend(); plt.grid(True, ls='--', alpha=0.35)
    plt.title("异常值检测与去除"); plt.tight_layout(); plt.show()

# =========================
# 3) 异常值处理（Hampel + 插值）
# =========================
def hampel_filter_interp(x, window=5, k=3.0):
    x = np.asarray(x, float)
    n = len(x)
    y = x.copy()
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i-window), min(n, i+window+1)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg-med))+1e-12
        thr = 1.4826*mad*k
        if abs(x[i]-med) > thr:
            y[i] = med; mask[i] = True
    if mask.any():
        idx = np.arange(n)
        y[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return y, mask

# =========================
# 4) 要求①：两种降噪
# =========================
def frequency_domain_denoise(x, level, keep_coarse_details=1, pad_mode='symmetric'):
    """
    频域法：保留低频（可选保留最粗若干层细节），其余高频置零
    """
    aL, details = wavedec_lifting(x, level, pad_mode)
    new_details = [np.zeros_like(d) for d in details]
    for i in range(1, keep_coarse_details+1):
        new_details[-i] = details[-i].copy()  # 只保留最粗若干层
    x_denoised = waverec_lifting(aL, new_details, pad_mode)[:len(x)]
    return x_denoised

def _soft(x, t): 
    return np.sign(x)*np.maximum(np.abs(x)-t, 0.0)

def amplitude_domain_denoise(x, level, mode='soft', pad_mode='symmetric'):
    """
    幅值域阈值法：对细节系数做软/硬阈值
    """
    aL, details = wavedec_lifting(x, level, pad_mode)
    mad0 = np.median(np.abs(details[0]-np.median(details[0])))+1e-12
    sigma = mad0/0.6745
    N = len(x); t = sigma*np.sqrt(2*np.log(N))
    new_details=[]
    for j,d in enumerate(details):
        tj = t/(2**j)  # 多尺度阈值
        new_details.append(_soft(d,tj) if mode=='soft' else d*(np.abs(d)>=tj))
    x_denoised = waverec_lifting(aL,new_details,pad_mode)[:len(x)]
    return x_denoised

# =========================
# 5) 要求② & ③：分离
# =========================
def method_highpass(x, level, pad_mode='symmetric'):
    """
    法①（高通）：令 s=0 -> 仅用各层细节重构出“高频”；低频= x - 高频
    """
    aL, details = wavedec_lifting(x, level, pad_mode)
    a_zero = np.zeros_like(aL)
    high = waverec_lifting(a_zero, details, pad_mode)[:len(x)]
    low  = x - high
    return low, high

def method_lowpass(x, level, pad_mode='symmetric'):
    """
    法②（低通）：令 d=0 -> 仅用近似系数重构出“低频”；高频= x - 低频
    """
    aL, details = wavedec_lifting(x, level, pad_mode)
    zero_details = [np.zeros_like(d) for d in details]
    low  = waverec_lifting(aL, zero_details, pad_mode)[:len(x)]
    high = x - low
    return low, high

# =========================
# 6) 主流程
# =========================
def run_pipeline(
    file_path,
    column_name,
    level=6,
    hampel_window=8,
    hampel_k=0.7,
    keep_coarse_details=1,
    pad_mode='symmetric'
):
    # 读取
    df = pd.read_excel(file_path)
    x_raw = df[column_name].astype(float).to_numpy()
    print(f"信号长度={len(x_raw)}, 分解层数={level}")

    # 去异常
    x_clean, mask = hampel_filter_interp(x_raw, window=hampel_window, k=hampel_k)
    plot_outlier_compare(x_raw, x_clean, mask)

    # 要求①：两种降噪
    x_freq = frequency_domain_denoise(x_clean, level, keep_coarse_details, pad_mode)
    x_amp  = amplitude_domain_denoise(x_clean, level, 'soft', pad_mode)
    plot_series(x_raw, {"频域降噪":x_freq}, "频域滤波法降噪结果")
    plot_series(x_raw, {"幅值域降噪":x_amp}, "幅值域滤波法降噪结果")
    plot_series(x_raw, {"频域降噪":x_freq,"幅值域降噪":x_amp}, "两种降噪结果对比")

    # 要求② & ③：两种分离（理论上两法结果应一致）
    low_h,  high_h  = method_highpass(x_clean, level, pad_mode)
    low_l,  high_l  = method_lowpass(x_clean, level, pad_mode)

    # 显示每法的分离
    plot_series(x_raw, {"法①-低频":low_h,"法①-高频":high_h}, "法①：轮廓(低频) + 粗糙度(高频)")
    plot_series(x_raw, {"法②-低频":low_l,"法②-高频":high_l}, "法②：轮廓(低频) + 粗糙度(高频)")

    # 法① vs 法② 对比（含原始）
    plot_series(x_raw, {"法①-轮廓(低频)": low_h, "法②-轮廓(低频)": low_l},
                "法①与法② 轮廓(低频) 对比（含原始）")
    plot_series(x_raw, {"法①-粗糙度(高频)": high_h, "法②-粗糙度(高频)": high_l},
                "法①与法② 粗糙度(高频) 对比（含原始）")

    # ——一致性校验（应非常小，~1e-10 量级，数值误差）
    low_diff  = np.max(np.abs(low_h  - low_l))
    high_diff = np.max(np.abs(high_h - high_l))
    print(f"[一致性校验] max|低频(法①-法②)| = {low_diff:.3e}")
    print(f"[一致性校验] max|高频(法①-法②)| = {high_diff:.3e}")

    return dict(
        x_raw=x_raw, x_clean=x_clean,
        freq_denoised=x_freq, amp_denoised=x_amp,
        highpass=dict(low=low_h, high=high_h),
        lowpass=dict(low=low_l, high=high_l),
        diffs=dict(low=low_diff, high=high_diff)
    )

# =========================
# 7) 运行配置
# =========================
if __name__ == "__main__":
    CONFIG = {
        "file_path": r"E:\学习资源！！！！！！\数字制造技术\task5\sine_noised2.xlsx",
        "column_name": "Noised data3",
        "level": 6,
        "hampel_window": 8,
        "hampel_k": 0.7,          # 根据数据可适当调大到 2~3
        "keep_coarse_details": 1, # 频域法：是否保留最粗细节层
        "pad_mode": "symmetric",
    }
    results = run_pipeline(**CONFIG)
