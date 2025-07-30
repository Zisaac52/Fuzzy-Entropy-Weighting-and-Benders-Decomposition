import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np

# 1. 数据准备 (与之前相同)
data_string = """
Method,Epochs,Accuracy
fedasync,1,32.5;fedasync,2,96.3;fedasync,3,97.4;fedasync,4,97.73;fedasync,5,97.83;fedasync,6,98.25;fedasync,7,98.38;fedasync,8,98.22;fedasync,9,98.46;fedasync,10,98.54
bafl,1,36.4;bafl,2,96.24;bafl,3,97.6;bafl,4,97.84;bafl,5,98.11;bafl,6,98.27;bafl,7,98.39;bafl,8,98.29;bafl,9,98.57;bafl,10,98.62
"fuzzy,m=5,r=0.7",1,37.56;"fuzzy,m=5,r=0.7",2,96.59;"fuzzy,m=5,r=0.7",3,97.16;"fuzzy,m=5,r=0.7",4,98.01;"fuzzy,m=5,r=0.7",5,98.17;"fuzzy,m=5,r=0.7",6,98.39;"fuzzy,m=5,r=0.7",7,98.54;"fuzzy,m=5,r=0.7",8,98.58;"fuzzy,m=5,r=0.7",9,98.67;"fuzzy,m=5,r=0.7",10,98.77
"fuzzy,m=3,r=0.2",1,51.05;"fuzzy,m=3,r=0.2",2,96.61;"fuzzy,m=3,r=0.2",3,97.52;"fuzzy,m=3,r=0.2",4,97.78;"fuzzy,m=3,r=0.2",5,98.19;"fuzzy,m=3,r=0.2",6,97.93;"fuzzy,m=3,r=0.2",7,98.33;"fuzzy,m=3,r=0.2",8,98.41;"fuzzy,m=3,r=0.2",9,98.48;"fuzzy,m=3,r=0.2",10,98.51
"fuzzy,m=6,r=0.2",1,44.89;"fuzzy,m=6,r=0.2",2,95.82;"fuzzy,m=6,r=0.2",3,97.33;"fuzzy,m=6,r=0.2",4,97.86;"fuzzy,m=6,r=0.2",5,97.97;"fuzzy,m=6,r=0.2",6,98.35;"fuzzy,m=6,r=0.2",7,98.32;"fuzzy,m=6,r=0.2",8,98.08;"fuzzy,m=6,r=0.2",9,98.41;"fuzzy,m=6,r=0.2",10,98.45
"fuzzy,m=5,r=0.2",1,42.92;"fuzzy,m=5,r=0.2",2,96.35;"fuzzy,m=5,r=0.2",3,97.74;"fuzzy,m=5,r=0.2",4,97.65;"fuzzy,m=5,r=0.2",5,98.2;"fuzzy,m=5,r=0.2",6,98.29;"fuzzy,m=5,r=0.2",7,98.27;"fuzzy,m=5,r=0.2",8,98.3;"fuzzy,m=5,r=0.2",9,98.32;"fuzzy,m=5,r=0.2",10,98.39
"fuzzy,m=5,r=0.5",1,42.92;"fuzzy,m=5,r=0.5",2,96.35;"fuzzy,m=5,r=0.5",3,97.74;"fuzzy,m=5,r=0.5",4,97.65;"fuzzy,m=5,r=0.5",5,98.2;"fuzzy,m=5,r=0.5",6,98.29;"fuzzy,m=5,r=0.5",7,98.27;"fuzzy,m=5,r=0.5",8,98.31;"fuzzy,m=5,r=0.5",9,98.33;"fuzzy,m=5,r=0.5",10,98.62
""".replace(';', '\n')
df = pd.read_csv(io.StringIO(data_string))

method_rename = {
    "fuzzy,m=5,r=0.7": "Fuzzy (Ours, m=5, r=0.7)", "fedasync": "FedAsync", "bafl": "BAFL",
    "fuzzy,m=3,r=0.2": "Fuzzy (m=3, r=0.2)", "fuzzy,m=6,r=0.2": "Fuzzy (m=6, r=0.2)",
    "fuzzy,m=5,r=0.2": "Fuzzy (m=5, r=0.2)", "fuzzy,m=5,r=0.5": "Fuzzy (m=5, r=0.5)"
}
df['Method'] = df['Method'].map(method_rename)

style_dict = {
    "Fuzzy (Ours, m=5, r=0.7)": {"color": "red", "linestyle": "-", "marker": "o", "linewidth": 2.5, "markersize": 7, "zorder": 10},
    "FedAsync": {"color": "blue", "linestyle": "--", "marker": "s", "linewidth": 1.5, "zorder": 5},
    "BAFL": {"color": "green", "linestyle": "-.", "marker": "^", "linewidth": 1.5, "zorder": 5},
    "Fuzzy (m=3, r=0.2)": {"color": "purple", "linestyle": ":", "marker": "d", "linewidth": 1.5, "alpha": 0.8},
    "Fuzzy (m=6, r=0.2)": {"color": "orange", "linestyle": ":", "marker": "v", "linewidth": 1.5, "alpha": 0.8},
    "Fuzzy (m=5, r=0.2)": {"color": "brown", "linestyle": ":", "marker": "x", "linewidth": 1.5, "alpha": 0.8},
    "Fuzzy (m=5, r=0.5)": {"color": "darkorange", "linestyle": ":", "marker": "+", "linewidth": 1.5, "alpha": 0.8},
}

# 2. 创建两个垂直排列的子图
# 核心修改：调整高度比例为[2, 1]，让上部图更高，视觉更平衡
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.1)

# 3. 在两个子图上都绘制所有曲线 (使用原始的 'Epochs' 列)
for method in df['Method'].unique():
    subset = df[df['Method'] == method]
    style = style_dict.get(method, {})
    ax_top.plot(subset['Epochs'], subset['Accuracy'], label=method, **style)
    ax_bottom.plot(subset['Epochs'], subset['Accuracy'], **style)

# 4. 设置Y轴范围
ax_top.set_ylim(95.5, 99.5)
ax_bottom.set_ylim(30, 55)

# 5. 隐藏不必要的坐标轴线和刻度
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False)

# 6. 添加断裂标记
d = .015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# 7. 添加标签、标题、图例和网格
fig.suptitle("Comparison of Convergence Performance", fontsize=16, weight='bold')
fig.text(0.06, 0.5, 'Test Accuracy (%)', va='center', rotation='vertical', fontsize=14)
ax_bottom.set_xlabel("Communication Rounds (Epochs)", fontsize=14)

# 将图例放入上部图的右下角空白处
ax_top.legend(loc='lower right', fontsize=11)

ax_top.grid(True, linestyle='--', alpha=0.6)
ax_bottom.grid(True, linestyle='--', alpha=0.6)
ax_bottom.set_xticks(np.arange(1, 11, 1))

# 8. 保存和显示
plt.savefig("fuzzy_comparison_final_adjusted_ratio.png", dpi=300, bbox_inches='tight')
plt.show()