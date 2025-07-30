import matplotlib.pyplot as plt
import numpy as np

# --- 数据和标签 ---
x_labels = ['2', '5', '8', '10', '15', '20', '25', '30']
x_pos = np.arange(len(x_labels))
component_labels = ['Master Problem Time', 'Subproblem Time', 'Relaxed Subproblem Time']
total_time_label = 'Total Time'

component_data = {
    component_labels[0]: np.array([0.03, 0.07, 0.03, 0.06, 0.04, 0.07, 0.13, 0.11]),
    component_labels[1]: np.array([0.28, 0.22, 0.29, 0.37, 0.54, 0.54, 0.72, 1.00]),
    component_labels[2]: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.22, 0.13])
}
total_time_data = np.array([0.32, 0.33, 0.34, 0.46, 0.61, 0.80, 1.15, 1.30])

# --- 绘图设置 ---
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except RuntimeError:
    plt.rcParams['font.family'] = 'serif'
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(8, 5))

# --- 绘制堆叠柱状图 ---
bar_width = 0.6
# 修正后的颜色获取方法
cmap = plt.get_cmap('BuGn') 
# 我们需要3种颜色，从色谱的30%到90%之间均匀选取，以避免太浅或太深的颜色
colors = cmap(np.linspace(0.3, 0.9, len(component_labels)))

bottom_y = np.zeros(len(x_labels))

for i, label in enumerate(component_labels):
    values = component_data[label]
    ax.bar(x_pos, values, bar_width, label=label, bottom=bottom_y, color=colors[i])
    bottom_y += values # 累加高度

# --- 绘制总时间作为对比线 ---
ax.plot(x_pos, total_time_data, marker='o', linestyle='--', color='black', label=total_time_label, alpha=0.7)

# --- 美化图表 ---
ax.set_xlabel('Number of Tasks', fontsize=12)
ax.set_ylabel('Solution Time (s)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylim(0, 1.5)

ax.legend(loc='upper left', fontsize=10)
ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='lightgray')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('stacked_bar_chart_fixed.pdf')
plt.show()