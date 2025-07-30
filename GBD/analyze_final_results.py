import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results_df, output_dir='results'):
    """
    根据分析结果绘制图表。

    参数:
    results_df (pd.DataFrame): 包含平均性能指标的DataFrame。
    output_dir (str): 保存图表的目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_theme(style="whitegrid")

    metrics_to_plot = {
        'total_time': 'Average Total Time (s)',
        'total_energy': 'Average Total Energy',
        'total_utility': 'Average Total Utility',
        'final_objective_ub': 'Average Final Objective (Zeta)'
    }

    for metric, ylabel in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        
        plot = sns.lineplot(data=results_df, x='num_clients', y=metric, hue='solver', marker='o', linestyle='-')
        
        plt.title(f'{ylabel} vs. Number of Clients', fontsize=16)
        plt.xlabel('Number of Clients', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(title='Solver Strategy')
        plt.grid(True)
        
        chart_path = os.path.join(output_dir, f'fairness_{metric}_analysis.png')
        plt.savefig(chart_path)
        plt.close()
        print(f"图表已保存到: {chart_path}")

def analyze_final_results(log_file='fairness_experiment_log.csv', output_file='docs/final_fairness_analysis_results.md'):
    """
    解析并分析最终的实验日志文件，计算平均指标，保存文本结果并绘制图表。

    参数:
    log_file (str): 实验日志文件的路径。
    output_file (str): 保存分析结果的Markdown文件路径。
    """
    try:
        # 加载数据
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"错误: 日志文件 '{log_file}' 未找到。")
        return None

    # 替换无穷大值以便计算
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 定义要计算平均值的指标
    metrics_to_average = [
        'total_time',
        'total_energy',
        'total_utility',
        'final_objective_ub'
    ]

    # 数据分组并计算平均指标
    results_df = df.groupby(['solver', 'num_clients'])[metrics_to_average].mean().reset_index()

    # 结果处理：将NaN填充为0或适当的值，这对于绘图很重要
    results_df.fillna(0, inplace=True)
    
    # 将结果存储在嵌套字典中以便于结构化输出
    structured_results = {}
    for _, row in results_df.iterrows():
        solver = row['solver']
        num_clients = int(row['num_clients'])
        
        if solver not in structured_results:
            structured_results[solver] = {}
        
        structured_results[solver][num_clients] = {
            'avg_total_time': row['total_time'],
            'avg_total_energy': row['total_energy'],
            'avg_total_utility': row['total_utility'],
            'avg_final_objective_ub (zeta)': row['final_objective_ub']
        }

    # 准备输出内容
    output_lines = []
    output_lines.append("# 实验结果最终分析\n\n")

    # 按策略和客户端数量排序后输出
    for solver in sorted(structured_results.keys()):
        output_lines.append(f"## 策略: {solver}\n")
        # 对客户端数量进行排序
        for num_clients in sorted(structured_results[solver].keys()):
            metrics = structured_results[solver][num_clients]
            output_lines.append(f"### 规模 (客户端数量): {num_clients}\n")
            output_lines.append(f"- **平均完成时间 (total_time):**       `{metrics['avg_total_time']:.4f}` s\n")
            output_lines.append(f"- **平均总能耗 (total_energy):**       `{metrics['avg_total_energy']:.4f}`\n")
            output_lines.append(f"- **平均总效用 (total_utility):**      `{metrics['avg_total_utility']:.4f}`\n")
            output_lines.append(f"- **平均最小系统效用 (zeta):** `{metrics['avg_final_objective_ub (zeta)']:.4f}`\n")
        output_lines.append("\n")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"分析结果已成功保存到: {output_file}")
    
    return results_df


if __name__ == '__main__':
    results = analyze_final_results()
    if results is not None:
        plot_results(results)