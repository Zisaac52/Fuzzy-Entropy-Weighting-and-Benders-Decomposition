import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

def load_and_clean_data(filepath="experiment_results_log.csv"):
    """
    读取CSV数据，并进行初步清洗。
    - 确保数值类型正确
    - 处理可能存在的非有限数值
    """
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None
        
    df = pd.read_csv(filepath)
    # 将列名中的空格去除
    df.columns = df.columns.str.strip()
    # 确保关键列是数值类型，处理错误
    for col in ['total_time', 'final_objective_ub', 'num_clients', 'strong_ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['total_time', 'final_objective_ub'], inplace=True)
    return df

def calculate_offload_rate(row):
    """
    根据 solution_x_i 列计算卸载率。
    x_i = 1 表示客户端 i 卸载。
    """
    try:
        # 安全地将字符串 '{0: 1, 1: 0}' 转换成字典
        solution_dict = ast.literal_eval(row['solution_x_i'])
        num_offloaded = sum(1 for val in solution_dict.values() if val == 1)
        if row['num_clients'] > 0:
            return num_offloaded / row['num_clients']
        return 0
    except (ValueError, SyntaxError):
        # 如果解析失败，返回None或0
        return None

def plot_performance_comparison(df, instance_clients=20):
    """
    生成 Q1 的性能对比条形图。
    注意: 当前数据中只有 'gbd' solver，此函数为未来扩展预留。
    """
    df_filtered = df[df['num_clients'] == instance_clients]
    if df_filtered.empty or len(df_filtered['solver'].unique()) < 2:
        print(f"Skipping performance comparison: Not enough data or solver variety for {instance_clients} clients.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_filtered, x='solver', y='final_objective_ub', estimator='mean', errorbar='sd')
    plt.title(f'Performance Comparison (num_clients={instance_clients})')
    plt.ylabel('Average Final Objective (UB)')
    plt.xlabel('Solver')
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png')
    plt.close()

def plot_scalability(df):
    """
    生成 Q2 的可扩展性线图。
    """
    # 只针对 'gbd' solver 进行分析
    df_gbd = df[df['solver'] == 'gbd'].copy()
    
    # 按客户端数量分组，计算平均时间和目标值
    scalability_data = df_gbd.groupby('num_clients').agg(
        avg_time=('total_time', 'mean'),
        avg_objective=('final_objective_ub', 'mean')
    ).reset_index()

    # 绘制时间可扩展性图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scalability_data, x='num_clients', y='avg_time', marker='o')
    plt.title('Scalability: Solve Time vs. Number of Clients (GBD)')
    plt.xlabel('Number of Clients')
    plt.ylabel('Average Total Solve Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/scalability_time.png')
    plt.close()
    print("Generated results/scalability_time.png")

    # 绘制目标值可扩展性图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scalability_data, x='num_clients', y='avg_objective', marker='o')
    plt.title('Scalability: Objective Value vs. Number of Clients (GBD)')
    plt.xlabel('Number of Clients')
    plt.ylabel('Average Final Objective (UB)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/scalability_objective.png')
    plt.close()
    print("Generated results/scalability_objective.png")


def plot_parameter_sensitivity(df):
    """
    生成 Q3 的参数敏感性分析图。
    """
    df_gbd = df[df['solver'] == 'gbd'].copy()
    
    # 计算卸载率并添加到DataFrame
    df_gbd['offload_rate'] = df_gbd.apply(calculate_offload_rate, axis=1)
    df_gbd.dropna(subset=['offload_rate'], inplace=True)
    
    if df_gbd.empty:
        print("Skipping parameter sensitivity plot: No data available after processing.")
        return

    # 按 strong_ratio 分组，计算平均卸载率
    sensitivity_data = df_gbd.groupby('strong_ratio')['offload_rate'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sensitivity_data, x='strong_ratio', y='offload_rate', marker='o')
    plt.title('Parameter Sensitivity: Offload Rate vs. Strong Client Ratio (GBD)')
    plt.xlabel('Ratio of Strong Clients')
    plt.ylabel('Average Offload Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/sensitivity_offload_rate.png')
    plt.close()
    print("Generated results/sensitivity_offload_rate.png")

def main():
    """
    主函数，协调整个分析和绘图流程。
    """
    # 确保结果目录存在
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. 加载数据
    df = load_and_clean_data("experiment_results_log.csv")

    if df is not None:
        # 2. 生成图表
        print("--- Generating Plots ---")
        
        # Q1: 性能对比 (为未来准备)
        plot_performance_comparison(df)
        
        # Q2: 可扩展性分析
        plot_scalability(df)
        
        # Q3: 参数敏感性分析
        plot_parameter_sensitivity(df)

        print("\nAll plots have been generated in the 'results/' directory.")
    else:
        print("Could not generate plots due to data loading issues.")

if __name__ == "__main__":
    main()