import os
os.system('cls')
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Excel 数据绘图程序（固定路径版本）
# 作者: ChatGPT (Expert Prompt Engineer)
# 功能: 读取 Excel 第1列为 x，第2列为 y，自动绘制折线图
# ===============================

# 1️⃣ 在这里填写你的 Excel 文件完整路径（示例路径请替换！）
excel_path = r"E:\学习资源！！！！！！\数字制造技术\task7\sine_noised3.xlsx"  # ← 这里改成你的Excel文件路径

# 2️⃣ 检查文件是否存在
if not os.path.exists(excel_path):
    raise FileNotFoundError(f"❌ 找不到指定的文件路径: {excel_path}")

# 3️⃣ 读取 Excel 数据
df = pd.read_excel(excel_path)

# 检查至少有两列
if df.shape[1] < 2:
    raise ValueError("Excel 文件中至少需要两列数据！第1列为x，第2列为y。")

# 4️⃣ 获取 x, y 数据
x = df.iloc[:, 0]
y = df.iloc[:, 1]

# 5️⃣ 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=1, label='y(x)')

# 6️⃣ 图形美化
plt.title("Excel 数据绘图", fontsize=14)
plt.xlabel(df.columns[0], fontsize=12)
plt.ylabel(df.columns[1], fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# 7️⃣ 显示图形
plt.show()
