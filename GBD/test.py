import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import griddata

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300

# 准备数据
accuracy_data = {
    'FuzzyM': [2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7, 9],
    'FuzzyR': [0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5],
    'Accuracy': [95.91, 96.82, 96.64, 96.54, 96.69, 96.28, 96.74, 96.94, 96.68, 96.26, 96.04, 96.78, 97.03, 96.12, 96.68, 96.78, 96.56, 96.39, 97.01, 96.35, 96.36, 96.35, 96.59, 96.73, 96.46]
}

loss_data = {
    'FuzzyM': [2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7, 9, 2, 3, 5, 7, 9],
    'FuzzyR': [0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5],
    'Loss': [0.2898, 0.1782, 0.2297, 0.2201, 0.2303, 0.3068, 0.2361, 0.1965, 0.1939, 0.2993, 0.2977, 0.2383, 0.1828, 0.2869, 0.2096, 0.1891, 0.244, 0.2359, 0.2199, 0.2129, 0.2398, 0.2995, 0.1854, 0.2005, 0.2301]
}

# 转换为DataFrame
df_accuracy = pd.DataFrame(accuracy_data)
df_loss = pd.DataFrame(loss_data)

# 创建网格数据
X = np.linspace(df_accuracy['FuzzyM'].min(), df_accuracy['FuzzyM'].max(), 100)
Y = np.linspace(df_accuracy['FuzzyR'].min(), df_accuracy['FuzzyR'].max(), 100)
X, Y = np.meshgrid(X, Y)

# 插值生成Z值（修正Loss的插值参数）
Z_accuracy = griddata((df_accuracy['FuzzyM'], df_accuracy['FuzzyR']), df_accuracy['Accuracy'], (X, Y), method='cubic')
Z_loss = griddata((df_loss['FuzzyM'], df_loss['FuzzyR']), df_loss['Loss'], (X, Y), method='cubic')

# 创建图形
fig = plt.figure(figsize=(16, 7))

# Accuracy三维曲面图
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_accuracy, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title('Relationship between Accuracy, FuzzyM, and FuzzyR', pad=20)
ax1.set_xlabel('FuzzyM', labelpad=10)
ax1.set_ylabel('FuzzyR', labelpad=10)
ax1.set_zlabel('Accuracy (%)', labelpad=10)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Accuracy Value')

# Loss三维曲面图
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_loss, cmap='plasma', edgecolor='none', alpha=0.8)
ax2.set_title('Relationship between Loss, FuzzyM, and FuzzyR', pad=20)
ax2.set_xlabel('FuzzyM', labelpad=10)
ax2.set_ylabel('FuzzyR', labelpad=10)
ax2.set_zlabel('Loss', labelpad=10)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label='Loss Value')

# 调整视角（优化观察角度）
ax1.view_init(elev=30, azim=45)
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig('plot.png')
print("Plot saved to plot.png")