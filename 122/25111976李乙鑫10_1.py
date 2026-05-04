import numpy as np
from itertools import product

# ============================================================
# 0. 在这里填入你的 8 个实验结果 y1~y8 （对应完整 2^3 设计）
#    推荐顺序：A,B,C 从 (-1,-1,-1) 递增到 (+1,+1,+1)
# ============================================================
# 例如：y = np.array([y1, y2, ..., y8], dtype=float)
y = np.array([
    # TODO: 把下面 8 个数字换成你自己的实验数据
    10.0, 11.0, 9.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
], dtype=float)

# ============================================================
# 1. 构造 2^3 全因子设计：A,B,C ∈ {-1, +1}
#    并写出联立方程所对应的设计矩阵 X_full
#    模型：y = w0 + wA*A + wB*B + wC*C
#              + wAB*A*B + wAC*A*C + wBC*B*C + wABC*A*B*C
# ============================================================

# 所有组合：(-1,-1,-1) ... (+1,+1,+1)，共 8 组
levels = [-1, 1]
design = np.array(list(product(levels, repeat=3)), dtype=float)  # shape (8,3)
A = design[:, 0]
B = design[:, 1]
C = design[:, 2]

# 全模型的设计矩阵 X_full（8×8）
X_full = np.column_stack([
    np.ones_like(A),   # w0 截距
    A,                 # wA
    B,                 # wB
    C,                 # wC
    A * B,             # wAB
    A * C,             # wAC
    B * C,             # wBC
    A * B * C          # wABC
])

print("=== 2^3 全因子设计的 (A,B,C) 组合 ===")
for i, (a, b, c) in enumerate(design, start=1):
    print(f"run {i}: A={a:+.0f}, B={b:+.0f}, C={c:+.0f},  对应方程："
          f"w0 + wA*({a:+.0f}) + wB*({b:+.0f}) + wC*({c:+.0f}) "
          f"+ wAB*({a*b:+.0f}) + wAC*({a*c:+.0f}) + wBC*({b*c:+.0f}) "
          f"+ wABC*({a*b*c:+.0f}) = y{i}")

# ============================================================
# 2. 利用 y_i 解上述方程中的所有 w（8 个系数）
#    因为是 8 个方程 8 个未知数，可以直接线性求解
# ============================================================
# w = [w0, wA, wB, wC, wAB, wAC, wBC, wABC]^T
w_full = np.linalg.solve(X_full, y)

w0, wA, wB, wC, wAB, wAC, wBC, wABC = w_full

print("\n=== 解出的全模型系数（包含主效应 + 交互作用） ===")
print(f"w0   = {w0:.6f}")
print(f"wA   = {wA:.6f}")
print(f"wB   = {wB:.6f}")
print(f"wC   = {wC:.6f}")
print(f"wAB  = {wAB:.6f}")
print(f"wAC  = {wAC:.6f}")
print(f"wBC  = {wBC:.6f}")
print(f"wABC = {wABC:.6f}")

# ============================================================
# 3. 只关心三个主效应：wA, wB, wC
#    模型：y = w0 + wA*A + wB*B + wC*C
#    这时只有 4 个未知数，8 组数据 → 过定约束 → 用最小二乘
# ============================================================
X_main = np.column_stack([
    np.ones_like(A),  # w0
    A,                # wA
    B,                # wB
    C                 # wC
])

# 最小二乘求解：w_main = argmin ||X_main w - y||^2
w_main, *_ = np.linalg.lstsq(X_main, y, rcond=None)
w0_m, wA_m, wB_m, wC_m = w_main

print("\n=== 只含主效应模型的系数（wA, wB, wC） ===")
print(f"w0 (main only) = {w0_m:.6f}")
print(f"wA (main only) = {wA_m:.6f}")
print(f"wB (main only) = {wB_m:.6f}")
print(f"wC (main only) = {wC_m:.6f}")

# 你可以用下面这两句话验证拟合效果：
y_pred_full = X_full @ w_full
y_pred_main = X_main @ w_main

print("\n=== 拟合残差（全模型 vs 只含主效应模型） ===")
print("全模型残差范数 ||y - y_pred_full|| =", np.linalg.norm(y - y_pred_full))
print("主效应模型残差范数 ||y - y_pred_main|| =", np.linalg.norm(y - y_pred_main))
