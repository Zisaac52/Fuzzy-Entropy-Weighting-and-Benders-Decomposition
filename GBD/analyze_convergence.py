import pandas as pd
import io

def analyze_gbd_runs(csv_path):
    """
    分析GBD综合运行日志，按客户端数量统计收敛情况。

    Args:
        csv_path (str): comprehensive_gbd_runs_log.csv文件的路径。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_path}' 未找到。")
        return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 按 'num_clients' 分组
    grouped = df.groupby('num_clients')

    print("==========================================================")
    print("      GBD 求解器在不同规模下的收敛性分析")
    print("==========================================================")
    print(f"{'客户端数量':<12} | {'总运行次数':<12} | {'收敛次数':<10} | {'失败次数':<10} | {'收敛成功率 (%)':<15}")
    print("----------------------------------------------------------")

    # 计算每个组的统计数据
    for num_clients, group in grouped:
        total_runs = len(group)
        converged_runs = group[group['status'] == 'Converged'].shape[0]
        failed_runs = total_runs - converged_runs
        convergence_rate = (converged_runs / total_runs) * 100 if total_runs > 0 else 0
        
        print(f"{str(num_clients).ljust(12)} | {str(total_runs).ljust(12)} | {str(converged_runs).ljust(10)} | {str(failed_runs).ljust(10)} | {convergence_rate:.2f}")

    print("==========================================================")

if __name__ == '__main__':
    # 假设CSV文件在当前目录下
    analyze_gbd_runs('comprehensive_gbd_runs_log.csv')