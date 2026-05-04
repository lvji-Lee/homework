import os
os.system('cls')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def read_two_column_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    suf = p.suffix.lower()
    if suf in [".csv", ".txt"]:
        # 尝试常见编码
        for enc in ("utf-8", "gbk", "latin1"):
            try:
                df = pd.read_csv(p, header=None, encoding=enc)
                return df
            except Exception:
                pass
        # 最后再抛错
        raise ValueError("无法以常见编码读取 CSV，请确认文件格式和编码。")
    elif suf in [".xls", ".xlsx", ".xlsm"]:
        return pd.read_excel(p, header=None)
    else:
        # 兜底尝试 csv
        return pd.read_csv(p, header=None)

def fit_circle_algebraic(x, y):
    # 求解 x^2 + y^2 + D x + E y + F = 0 的最小二乘
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = C
    x0 = -D / 2
    y0 = -E / 2
    r = np.sqrt((D**2 + E**2) / 4 - F)
    return x0, y0, r

def plot_results(x, y, x0, y0, r, residuals, save_prefix=None):
    # 1) 原始点 + 拟合圆
    theta = np.linspace(0, 2*np.pi, 800)
    xc = x0 + r * np.cos(theta)
    yc = y0 + r * np.sin(theta)

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, label="Original Data", s=20,color='blue')
    plt.plot(xc, yc, label="Fitted Circle", linewidth=2,color='red')
    plt.scatter([x0], [y0], color="black", label="Center")
    plt.gca().set_aspect('equal', 'box')
    plt.title("Circle Fitting Result")
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_fit_circle.png", dpi=200, bbox_inches="tight")
    plt.show()

    # 2) 残差随点序号变化
    plt.figure(figsize=(8,4))
    plt.plot(residuals, marker='o', linestyle='-')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Residuals fingure")
    plt.xlabel("Index")
    plt.ylabel("Errors")
    if save_prefix:
        plt.savefig(f"{save_prefix}_residuals_index.png", dpi=200, bbox_inches="tight")
    plt.show()

    # 3) 残差空间分布（散点图，用颜色表示残差）
    plt.figure(figsize=(6,6))
    sc = plt.scatter(x, y, c=residuals, cmap='bwr', s=35)
    plt.colorbar(sc, label='Errors')
    plt.scatter([x0], [y0], color='k', marker='x', label='Center')
    plt.title("Spatial Residuals ")
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_residuals_spatial.png", dpi=200, bbox_inches="tight")
    plt.show()

def main(file_path):
    # 文件路径：推荐使用 Path 或原始字符串 r"..." 或者用正斜杠 /
    print("读取文件：", file_path)
    df = read_two_column_file(file_path)
    if df.shape[1] < 2:
        raise ValueError("文件至少应包含两列数据（无表头）。")
    x = df.iloc[:,0].astype(float).values
    y = df.iloc[:,1].astype(float).values

    x0, y0, r = fit_circle_algebraic(x, y)
    print("拟合结果：")
    print(f"  圆心 x0 = {x0:.6f}")
    print(f"  圆心 y0 = {y0:.6f}")
    print(f"  半径 r  = {r:.6f}")

    distances = np.sqrt((x - x0)**2 + (y - y0)**2)
    residuals = distances - r
    # 基本统计
    print("残差统计 :")
    print(f"  mean = {np.mean(residuals):.6e}")
    print(f"  std  = {np.std(residuals):.6e}")
    print(f"  max  = {np.max(residuals):.6e}")
    print(f"  min  = {np.min(residuals):.6e}")

    # 绘图并保存（可选）
    save_prefix = None
    # save_prefix = "circle_fit_output"   # 若想保存图片，取消注释并设置前缀
    plot_results(x, y, x0, y0, r, residuals, save_prefix=save_prefix)

if __name__ == "__main__":
    # ---------- 在这里把你的文件路径放好 ----------
    # 推荐写法 1：原始字符串 r"..."
    # file_path = r"E:\学习资源！！！！！！\数字制造技术\task9\circle_fitting_data (1).csv"
    # 推荐写法 2：用正斜杠（更稳健）
    # file_path = "E:/学习资源！！！！！！/数字制造技术/task9/circle_fitting_data (1).csv"
    # 推荐写法 3：把文件放到脚本同目录下然后只写文件名
    file_path = r"E:\学习资源！！！！！！\数字制造技术\task9\工作簿1.xlsx"

    try:
        main(file_path)
    except Exception as e:
        print("运行出错：", str(e))
        # 打印更详细的异常信息
        import traceback
        traceback.print_exc()
        sys.exit(1)

