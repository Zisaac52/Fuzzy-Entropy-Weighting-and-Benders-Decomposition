import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec # 导入GridSpec

# --- 1. 数据准备 (省略) ---
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
# --- 2. 插值数据准备 (省略) ---
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
# ================= 全局字体和大小设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 21
plt.rcParams['axes.labelsize'] = 23
plt.rcParams['axes.titlesize'] = 25
# =====================================================

# 调整整张大图的尺寸
fig = plt.figure(figsize=(20, 16))
# 调整总标题的内容和字体大小
fig.suptitle('Impact of Fuzzy Entropy Parameters (M and R) on Model Performance', fontsize=24, y=0.98)

# 使用 GridSpec 定义布局
gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig, width_ratios=[2, 1.5, 1.5])

# --- (a) Accuracy Heatmap ---
ax1 = fig.add_subplot(gs[0, 0])
# 【核心修改】在 cbar_kws 中加入 'shrink' 参数，控制颜色条的高度
# 0.77 是一个经验值，使颜色条高度与方形热力图视觉上匹配
sns.heatmap(accuracy_df, annot=True, fmt=".2f", cmap="viridis", ax=ax1,
            cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.84})
ax1.set_title('(a) Accuracy Heatmap')
ax1.set_xlabel('FuzzyR')
ax1.set_ylabel('FuzzyM')
# 强制热力图的单元格为方形
ax1.set_aspect('equal')

# --- (b) Smoothed Accuracy 3D Surface ---
ax2 = fig.add_subplot(gs[0, 1:], projection='3d')
surf2 = ax2.plot_surface(R_grid, M_grid, Z_acc_smooth, cmap='viridis', edgecolor='none', alpha=0.9)
ax2.set_title('(b) Smoothed Accuracy Surface Plot')
ax2.set_xlabel('FuzzyR', labelpad=20)
ax2.set_ylabel('FuzzyM', labelpad=20)
ax2.set_zlabel('Accuracy (%)', labelpad=20)
ax2.tick_params(axis='z', pad=10)
fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20, pad=0.15)
# 统一3D图的视角，确保坐标轴一致
ax2.view_init(elev=25, azim=-65)

# --- (c) Loss Heatmap ---
ax3 = fig.add_subplot(gs[1, 0])
# 【核心修改】同样为Loss热力图的颜色条设置 'shrink' 参数
sns.heatmap(loss_df, annot=True, fmt=".4f", cmap="OrRd", ax=ax3,
            cbar_kws={'label': 'Loss Value', 'shrink': 0.84})
ax3.set_title('(c) Loss Heatmap')
ax3.set_xlabel('FuzzyR')
ax3.set_ylabel('FuzzyM')
# 强制热力图的单元格为方形
ax3.set_aspect('equal')

# --- (d) Smoothed Loss 3D Surface ---
ax4 = fig.add_subplot(gs[1, 1:], projection='3d')
surf4 = ax4.plot_surface(R_grid, M_grid, Z_loss_smooth, cmap='OrRd', edgecolor='none', alpha=0.9)
ax4.set_title('(d) Smoothed Loss Surface Plot')
ax4.set_xlabel('FuzzyR', labelpad=20)
ax4.set_ylabel('FuzzyM', labelpad=20)
ax4.set_zlabel('Loss Value', labelpad=20)
ax4.tick_params(axis='z', pad=10)
fig.colorbar(surf4, ax=ax4, shrink=0.6, aspect=20, pad=0.15)
# 统一3D图的视角，与上方的图保持一致
ax4.view_init(elev=25, azim=-65)

# --- 4. 调整布局并保存 ---
fig.tight_layout(rect=[0, 0, 1, 1])
# plt.subplots_adjust(wspace=0.1)

plt.savefig('Fuzzy_Entropy_Parameter_Analysis_Final_Corrected_Cbar.png', dpi=300, bbox_inches='tight')
plt.show()
