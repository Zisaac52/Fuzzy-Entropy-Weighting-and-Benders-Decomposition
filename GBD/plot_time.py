import matplotlib.pyplot as plt
import numpy as np

# 1. 更新后的数据 (包括新增的 'all_offload')
x_data = np.array([2, 5, 8, 10, 15, 20, 25, 30])

# --- 原始数据 (根据您的新图微调) ---
gbd_y = np.array([0.32, 0.34, 0.35, 0.47, 0.61, 0.81, 1.14, 1.30])
local_first_y = np.array([0.58, 0.88, 1.20, 1.47, 1.82, 2.00, 2.25, 2.32])
# 注意: 'newgreedy' 在新图中为 'greedy'
greedy_y = np.array([0.46, 0.70, 0.88, 0.97, 1.18, 1.28, 1.35, 1.48])
random_offload_y = np.array([0.73, 1.21, 1.55, 1.75, 1.80, 1.95, 1.90, 1.93])
# --- 新增数据 ---
all_offload_y = np.array([0.45, 0.70, 0.75, 0.90, 1.08, 1.30, 1.60, 1.95])

# 2. 开始绘图，设置专业风格
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except RuntimeError:
    plt.rcParams['font.family'] = 'serif'

plt.style.use('seaborn-v0_8-paper')
plt.figure(figsize=(6, 4.5))

# 3. 更新样式列表以容纳5条线
# Matplotlib的'tab10'色板是公认的优秀色板，非常适合多条线
colors = plt.get_cmap('tab10').colors
markers = ['o', 's', '^', 'D', 'v'] # 新增倒三角 'v'
linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))] # 新增自定义点划线
labels = ['GBD', 'Local First', 'Greedy', 'Random Offload', 'All Offload']
data_series = [gbd_y, local_first_y, greedy_y, random_offload_y, all_offload_y]

# 4. 循环绘制五条线
for i in range(len(data_series)):
    plt.plot(x_data, data_series[i],
             label=labels[i],
             color=colors[i],
             marker=markers[i],
             linestyle=linestyles[i],
             linewidth=1.5,
             markersize=6)

# 5. 设置坐标轴、图例和网格
ax = plt.gca()
ax.set_xlabel('Number of Tasks', fontsize=12)
ax.set_ylabel('Average Completion Time (s)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(0, 35)
ax.set_ylim(0, 2.6)

# 图例现在有5项，可以调整位置或大小，'best'通常能找到好位置
ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.8, edgecolor='white')

ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# 6. 优化布局并保存
plt.tight_layout()
plt.savefig('beautified_chart_5_lines.pdf', format='pdf', dpi=300)
plt.savefig('beautified_chart_5_lines.png', format='png', dpi=300)
plt.show()