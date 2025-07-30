import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# --- 1. 数据准备 (与之前相同) ---
accuracy_data = {
    0.2: [95.91, 96.82, 96.64, 96.54, 96.69], 0.5: [96.28, 96.74, 96.94, 96.68, 96.26],
    0.7: [96.04, 96.78, 97.03, 96.12, 96.68], 1.0: [96.78, 96.56, 96.39, 97.01, 96.35],
    1.5: [96.36, 96.35, 96.59, 96.73, 96.46]
}
accuracy_df = pd.DataFrame(accuracy_data, index=[2, 3, 5, 7, 9])
loss_data = {
    0.2: [0.2898, 0.1782, 0.2297, 0.2201, 0.2303], 0.5: [0.3068, 0.2361, 0.1965, 0.1939, 0.2993],
    0.7: [0.2977, 0.2383, 0.1828, 0.2869, 0.2096], 1.0: [0.1891, 0.2440, 0.2359, 0.2199, 0.2129],
    1.5: [0.2398, 0.2995, 0.1854, 0.2005, 0.2301]
}
loss_df = pd.DataFrame(loss_data, index=[2, 3, 5, 7, 9])

# --- 2. 插值数据准备 (与之前相同) ---
R_orig, M_orig = np.meshgrid(accuracy_df.columns, accuracy_df.index)
points = np.array([R_orig.ravel(), M_orig.ravel()]).T
values_acc = accuracy_df.values.ravel()
values_loss = loss_df.values.ravel()
R_grid, M_grid = np.meshgrid(
    np.linspace(accuracy_df.columns.min(), accuracy_df.columns.max(), 100),
    np.linspace(accuracy_df.index.min(), accuracy_df.index.max(), 100)
)
Z_acc_smooth = griddata(points, values_acc, (R_grid, M_grid), method='cubic')
Z_loss_smooth = griddata(points, values_loss, (R_grid, M_grid), method='cubic')

# --- 3. 绘图设置 ---
# 全局字体和大小设置，对所有图生效
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 21
plt.rcParams['axes.labelsize'] = 23
plt.rcParams['axes.titlesize'] = 25


# ==========================================================
# 图1: Accuracy Heatmap
# ==========================================================
print("Generating Accuracy Heatmap...")
# 为单个图创建独立的画布和坐标轴，figsize可以根据需要调整
fig, ax = plt.subplots(figsize=(10, 8))
# 当子图区域本身接近方形时，shrink=1即可让颜色条与图等高
sns.heatmap(accuracy_df, annot=True, fmt=".2f", cmap="viridis", ax=ax,
            cbar_kws={'label': 'Accuracy (%)', 'shrink': 1.0})
ax.set_aspect('equal') # 强制单元格为方形
ax.set_xlabel('FuzzyR')
ax.set_ylabel('FuzzyM')
# 将标题放置在图的下方
ax.text(0.5, -0.2, '(a) Accuracy Heatmap', transform=ax.transAxes,
         ha='center', va='top', fontsize=plt.rcParams['axes.titlesize'])
plt.savefig('accuracy_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig) # 关闭当前图形，防止在内存中叠加


# ==========================================================
# 图2: Accuracy 3D Surface Plot
# ==========================================================
print("Generating Accuracy 3D Surface Plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R_grid, M_grid, Z_acc_smooth, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('FuzzyR', labelpad=20)
ax.set_ylabel('FuzzyM', labelpad=20)
ax.set_zlabel('Accuracy (%)', labelpad=20)
ax.tick_params(axis='z', pad=10)
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
ax.view_init(elev=25, azim=-65)
# 将标题放置在图的下方 (使用 text2D)
ax.text2D(0.5, 0.0, '(b) Smoothed Accuracy Surface Plot', transform=ax.transAxes,
         ha='center', va='top', fontsize=plt.rcParams['axes.titlesize'])
plt.savefig('accuracy_surface.png', dpi=300, bbox_inches='tight')
plt.close(fig)


# ==========================================================
# 图3: Loss Heatmap
# ==========================================================
print("Generating Loss Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(loss_df, annot=True, fmt=".4f", cmap="OrRd", ax=ax,
            cbar_kws={'label': 'Loss Value', 'shrink': 1.0})
ax.set_aspect('equal')
ax.set_xlabel('FuzzyR')
ax.set_ylabel('FuzzyM')
# 将标题放置在图的下方
ax.text(0.5, -0.2, '(c) Loss Heatmap', transform=ax.transAxes,
         ha='center', va='top', fontsize=plt.rcParams['axes.titlesize'])
plt.savefig('loss_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig)


# ==========================================================
# 图4: Loss 3D Surface Plot
# ==========================================================
print("Generating Loss 3D Surface Plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R_grid, M_grid, Z_loss_smooth, cmap='OrRd', edgecolor='none', alpha=0.9)
ax.set_xlabel('FuzzyR', labelpad=20)
ax.set_ylabel('FuzzyM', labelpad=20)
ax.set_zlabel('Loss Value', labelpad=20)
ax.tick_params(axis='z', pad=10)
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
ax.view_init(elev=25, azim=-65)
# 将标题放置在图的下方 (使用 text2D)
ax.text2D(0.5, 0.0, '(d) Smoothed Loss Surface Plot', transform=ax.transAxes,
         ha='center', va='top', fontsize=plt.rcParams['axes.titlesize'])
plt.savefig('loss_surface.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("\nAll four plots have been generated and saved successfully.")