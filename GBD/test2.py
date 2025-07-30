import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# --- 1. 数据准备 (与之前相同) ---
# ... (数据部分代码与之前完全一致，此处省略以保持简洁)
accuracy_data = {
    0.2: [95.91, 96.82, 96.64, 96.54, 96.69],
    0.5: [96.28, 96.74, 96.94, 96.68, 96.26],
    0.7: [96.04, 96.78, 97.03, 96.12, 96.68],
    1.0: [96.78, 96.56, 96.39, 97.01, 96.35],
    1.5: [96.36, 96.35, 96.59, 96.73, 96.46]
}
accuracy_df = pd.DataFrame(accuracy_data, index=[2, 3, 5, 7, 9])
accuracy_df.index.name = 'FuzzyM'
accuracy_df.columns.name = 'FuzzyR'

loss_data = {
    0.2: [0.2898, 0.1782, 0.2297, 0.2201, 0.2303],
    0.5: [0.3068, 0.2361, 0.1965, 0.1939, 0.2993],
    0.7: [0.2977, 0.2383, 0.1828, 0.2869, 0.2096],
    1.0: [0.1891, 0.2440, 0.2359, 0.2199, 0.2129],
    1.5: [0.2398, 0.2995, 0.1854, 0.2005, 0.2301]
}
loss_df = pd.DataFrame(loss_data, index=[2, 3, 5, 7, 9])
loss_df.index.name = 'FuzzyM'
loss_df.columns.name = 'FuzzyR'

# --- 2. 为3D图准备插值数据 (与之前相同) ---
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


# --- 3. 绘图设置 (附详细中文注释) ---
# ================= 全局字体和大小设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 【修改点】在这里调整全局基础字体大小
plt.rcParams['font.size'] = 16
# 【修改点】在这里调整坐标轴标签(如'FuzzyM')的字体大小
plt.rcParams['axes.labelsize'] = 18
# 【修改点】在这里调整子图标题(如'(a) Accuracy Heatmap')的字体大小
plt.rcParams['axes.titlesize'] = 20
# =====================================================

# 【修改点】在这里调整整张大图的尺寸 (宽度, 高度)，单位是英寸
fig = plt.figure(figsize=(18, 16))
# 【修改点】在这里调整总标题的内容和字体大小
fig.suptitle('Impact of Fuzzy Entropy Parameters (M and R) on Model Performance', fontsize=24, y=0.98)

# --- (a) Accuracy Heatmap ---
ax1 = fig.add_subplot(2, 2, 1)
sns.heatmap(accuracy_df, annot=True, fmt=".2f", cmap="viridis", ax=ax1,
            cbar_kws={'label': 'Accuracy (%)'})
ax1.set_title('(a) Accuracy Heatmap')
ax1.set_xlabel('FuzzyR')
ax1.set_ylabel('FuzzyM')

# --- (b) Smoothed Accuracy 3D Surface ---
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(R_grid, M_grid, Z_acc_smooth, cmap='viridis', edgecolor='none', alpha=0.9)
ax2.set_title('(b) Smoothed Accuracy Surface Plot')
# 【修改点】labelpad 用于调整坐标轴标签(如'FuzzyR')与坐标轴的距离
ax2.set_xlabel('FuzzyR', labelpad=15)
ax2.set_ylabel('FuzzyM', labelpad=15)
ax2.set_zlabel('Accuracy (%)', labelpad=15)
# 【核心修改】通过 tick_params 的 pad 参数，增加刻度数字与坐标轴的距离
ax2.tick_params(axis='x', pad=5)
ax2.tick_params(axis='y', pad=5)
ax2.tick_params(axis='z', pad=10)
# 设置颜色条
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20, pad=0.15)
cbar2.set_label('Accuracy (%)', size=16)
# 设置视角
ax2.view_init(elev=25, azim=-65)

# --- (c) Loss Heatmap ---
ax3 = fig.add_subplot(2, 2, 3)
sns.heatmap(loss_df, annot=True, fmt=".4f", cmap="OrRd", ax=ax3,
            cbar_kws={'label': 'Loss Value'})
ax3.set_title('(c) Loss Heatmap')
ax3.set_xlabel('FuzzyR')
ax3.set_ylabel('FuzzyM')

# --- (d) Smoothed Loss 3D Surface ---
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
surf4 = ax4.plot_surface(R_grid, M_grid, Z_loss_smooth, cmap='OrRd', edgecolor='none', alpha=0.9)
ax4.set_title('(d) Smoothed Loss Surface Plot')
# 【修改点】labelpad 用于调整坐标轴标签与坐标轴的距离
ax4.set_xlabel('FuzzyR', labelpad=15)
ax4.set_ylabel('FuzzyM', labelpad=15)
ax4.set_zlabel('Loss Value', labelpad=15)
# 【核心修改】通过 tick_params 的 pad 参数，增加刻度数字与坐标轴的距离
ax4.tick_params(axis='x', pad=5)
ax4.tick_params(axis='y', pad=5)
ax4.tick_params(axis='z', pad=10)
# 设置颜色条
cbar4 = fig.colorbar(surf4, ax=ax4, shrink=0.6, aspect=20, pad=0.15)
cbar4.set_label('Loss Value', size=16)
# 设置视角
ax4.view_init(elev=25, azim=115) 

# --- 4. 调整布局并保存 ---
# 【修改点】在这里调整子图之间的间距
# hspace 是子图之间的垂直间距
# wspace 是子图之间的水平间距
plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.1)
plt.savefig('Fuzzy_Entropy_Parameter_Analysis_Final_Adjusted.png', dpi=300, bbox_inches='tight')
plt.show()