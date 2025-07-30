# -*- coding: utf-8 -*-
"""
记录优化结果到 CSV 文件。
"""

import csv
import os

def log_results(filename, iteration_data, final_solution_summary, results_dir="results"):
    """
    将迭代数据和最终解决方案摘要记录到指定的 CSV 文件中。

    Args:
        filename (str): CSV 文件的名称 (不含扩展名)。
        iteration_data (list of dict): 每次迭代的数据列表。
            示例: [{'iteration': 1, 'lower_bound': 10.5, ...}, ...]
        final_solution_summary (dict): 最终解决方案的摘要信息。
            示例: {'total_time_s': 10.0, 'objective_value': 15.05, ...}
        results_dir (str): 存放结果文件的目录名称。
    """
    if not filename.endswith(".csv"):
        filename_csv = f"{filename}.csv"
    else:
        filename_csv = filename

    # 创建结果目录 (如果不存在)
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print(f"Created results directory: {results_dir}")
        except OSError as e:
            print(f"Error creating directory {results_dir}: {e}")
            # 如果无法创建目录，则记录在当前目录
            results_dir = "." 
    
    filepath = os.path.join(results_dir, filename_csv)

    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # 写入最终解决方案摘要
            if final_solution_summary:
                writer.writerow(["Metric", "Value"]) # 表头
                for key, value in final_solution_summary.items():
                    writer.writerow([key, value])
                writer.writerow([]) # 空行分隔

            # 写入迭代数据
            if iteration_data:
                # 从第一个迭代数据点获取表头 (假设所有数据点都有相同的键)
                # 如果 iteration_data 为空，则不执行
                header = list(iteration_data[0].keys())
                writer.writerow(header)
                for row_dict in iteration_data:
                    writer.writerow([row_dict.get(col, "") for col in header])
            
            print(f"Results successfully logged to: {filepath}")

    except IOError as e:
        print(f"Error writing to CSV file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while logging results: {e}")

if __name__ == '__main__':
    # 示例用法
    example_summary = {
        'total_time_s': 65.7,
        'total_iterations': 15,
        'final_lower_bound': 150.25,
        'final_upper_bound': 150.30,
        'final_gap_percentage': 0.033,
        'solution_status': 'Optimal (within gap)',
        'objective_value': 150.28,
        'num_clients': 10,
        'num_helpers': 5,
        'heterogeneity': 'medium'
    }

    example_iterations = [
        {'iteration': 1, 'lower_bound': 100.0, 'upper_bound': 200.0, 'time_s': 5.2, 'master_time_s': 0.5, 'subproblem_time_s': 4.5, 'cuts_added': 10, 'gap_perc': 100.0},
        {'iteration': 2, 'lower_bound': 120.0, 'upper_bound': 180.0, 'time_s': 4.8, 'master_time_s': 0.4, 'subproblem_time_s': 4.2, 'cuts_added': 8, 'gap_perc': 50.0},
        {'iteration': 3, 'lower_bound': 135.0, 'upper_bound': 160.0, 'time_s': 5.0, 'master_time_s': 0.5, 'subproblem_time_s': 4.3, 'cuts_added': 7, 'gap_perc': 18.5},
        # ... 更多迭代
        {'iteration': 15, 'lower_bound': 150.25, 'upper_bound': 150.30, 'time_s': 3.1, 'master_time_s': 0.3, 'subproblem_time_s': 2.7, 'cuts_added': 2, 'gap_perc': 0.033}
    ]
    
    # 测试1: 包含摘要和迭代数据
    log_results(
        filename="example_run_full",
        iteration_data=example_iterations,
        final_solution_summary=example_summary,
        results_dir="temp_results_test" # 使用临时目录进行测试
    )

    # 测试2: 仅包含摘要
    log_results(
        filename="example_run_summary_only",
        iteration_data=[],
        final_solution_summary=example_summary,
        results_dir="temp_results_test"
    )

    # 测试3: 仅包含迭代数据
    log_results(
        filename="example_run_iterations_only",
        iteration_data=example_iterations,
        final_solution_summary={}, # 或者 None
        results_dir="temp_results_test"
    )
    
    # 测试4: 空数据
    log_results(
        filename="example_run_empty",
        iteration_data=[],
        final_solution_summary=None, # 或者 {}
        results_dir="temp_results_test"
    )
    print(f"\nCheck the '{os.path.join(os.getcwd(), 'temp_results_test')}' directory for output files.")
    print("You may want to manually delete this directory after checking.")