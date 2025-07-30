import matplotlib.pyplot as plt
import numpy as np

# --- 数据和标签 ---
x_labels = ['2', '5', '8', '10', '15', '20', '25', '30']
x_pos = np.arange(len(x_labels))
labels = ['Total Time', 'Master Problem Time', 'Subproblem Time', 'Relaxed Subproblem Time']
data = {
    labels[0]: np.array([0.32, 0.33, 0.34, 0.46, 0.61, 0.80, 1.15, 1.30]),
    labels[1]: np.array([0.03, 0.07, 0.03, 0.06, 0.04, 0.07, 0.13, 0.11]),
    labels[2]: np.array([0.28, 0.22, 0.29, 0.37, 0.54, 0.54, 0.72, 1.00]),
    labels[3]: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.22, 0.13])
}

# --- 绘图设置 ---
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except RuntimeError:
    plt.rcParams['font.family'] = 'serif'
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(8, 5)) # 柱状图可以宽一些

# --- 绘制分组柱状图 ---
bar_width = 0.2
num_bars = len(data)
colors = plt.get_cmap('Paired').colors # 'Paired' 色板很适合柱状图

for i, (label, values) in enumerate(data.items()):
    # 计算每个柱子的位置
    offset = (i - num_bars / 2 + 0.5) * bar_width
    ax.bar(x_pos + offset, values, bar_width, label=label, color=colors[i*2+1])

# --- 美化图表 ---
ax.set_xlabel('Number of Tasks', fontsize=12)
ax.set_ylabel('Solution Time (s)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylim(0, 1.5)

ax.legend(loc='upper left', fontsize=10)
ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color='lightgray')
ax.set_axisbelow(True) # 让网格线在柱子后面

plt.tight_layout()
plt.savefig('grouped_bar_chart.pdf')
plt.show()