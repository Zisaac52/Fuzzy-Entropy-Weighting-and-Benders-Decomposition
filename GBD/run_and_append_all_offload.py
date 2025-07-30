import pandas as pd
import subprocess
import sys

def run_and_append_all_offload(log_file='fairness_experiment_log.csv'):
    """
    读取现有的实验日志，为其中每个 (num_clients, seed) 组合运行 'all_offload' 策略，
    并将结果追加到同一个日志文件中。
    """
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"错误: 日志文件 '{log_file}' 未找到。无法继续。")
        return

    # 找出所有成功的 (num_clients, seed, num_helpers) 组合
    all_successful_combinations = df[
        (df['solver'] != 'all_offload') &
        (df['status'].isin(['Converged', 'Optimal', 'Heuristic']))
    ][['num_clients', 'seed', 'num_helpers']].drop_duplicates()

    # 找出 'all_offload' 已经运行过的组合
    all_offload_runs = df[df['solver'] == 'all_offload'][['num_clients', 'seed', 'num_helpers']].drop_duplicates()

    # 使用merge来找到需要补充的组合
    # indicator=True 会添加一个 '_merge' 列，我们可以用它来筛选
    merged = pd.merge(all_successful_combinations, all_offload_runs, on=['num_clients', 'seed', 'num_helpers'], how='left', indicator=True)
    runs_to_add = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    if runs_to_add.empty:
        print("所有 'all_offload' 实验数据均已存在，无需补充。")
        return
        
    print(f"发现 {len(runs_to_add)} 个组合需要补充 'all_offload' 的实验数据。")
    
    # 对每个缺失的组合运行 'all_offload' 实验
    for index, row in runs_to_add.iterrows():
        num_clients = int(row['num_clients'])
        seed = int(row['seed'])
        num_helpers = int(row['num_helpers'])

        print("\n" + "="*50)
        print(f"正在为以下组合运行 'all_offload':")
        print(f"客户端数量: {num_clients}, 种子: {seed}, 辅助节点数量: {num_helpers}")
        print("="*50)

        # 构建命令行指令
        command = [
            sys.executable,  # 使用当前Python解释器
            'run_experiment.py',
            '--solver', 'all_offload',
            '--num_clients', str(num_clients),
            '--num_helpers', str(num_helpers),
            '--seed', str(seed),
            '--log_file', log_file
        ]
        
        # 执行指令
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("命令成功执行。输出:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败，错误码: {e.returncode}")
            print("标准输出:")
            print(e.stdout)
            print("标准错误:")
            print(e.stderr)
            print("-" * 50)
    
    print("\n所有 'all_offload' 实验已完成并追加到日志文件。")

if __name__ == '__main__':
    run_and_append_all_offload()