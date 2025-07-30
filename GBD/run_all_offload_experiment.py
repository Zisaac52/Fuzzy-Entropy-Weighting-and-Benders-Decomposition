# run_all_offload_experiment.py
# -*- coding: utf-8 -*-
"""
专门用于运行 'All Offload' 基准算法并记录结果的脚本。
"""
import csv
import os
import time
import random
from datetime import datetime

from gbd_resource_optimizer.data_utils.instance_generator import generate_instance
from gbd_resource_optimizer.baselines.all_offload import solve_all_offload

# --- 实验参数 ---
LOG_FILE = 'all_offload_runs_log.csv'
CLIENT_SCALES = [2, 5, 8, 10, 15, 20, 25, 30] # 与您的GBD实验保持一致的客户端规模
NUM_HELPERS = 2 # 与您的GBD实验保持一致的帮助者数量
SEEDS_PER_SCALE = 10 # 每个规模运行10次以获得平均性能

def setup_logger():
    """
    初始化CSV日志文件。
    每次运行时都重新创建文件并写入表头，以确保日志是全新的。
    """
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # 与 comprehensive_gbd_runs_log.csv 的表头保持一致
        writer.writerow([
            'timestamp', 'seed', 'num_clients', 'num_helpers', 'status',
            'final_lb', 'final_ub', 'total_time', 'solution_x',
            'total_energy', 'total_utility'
        ])
    print(f"Re-created log file '{LOG_FILE}' with headers.")

def run_single_experiment(num_clients, num_helpers, seed):
    """运行单次 all_offload 实验。"""
    print(f"\n--- Running All Offload: C={num_clients}, H={num_helpers}, Seed={seed} ---")
    
    # 1. 设置随机种子以保证问题实例的可复现性
    random.seed(seed)
    
    # 2. 生成问题实例 (与 GBD 使用相同的生成器)
    client_params, helper_params, network_params = generate_instance(num_clients, num_helpers)
    
    # 2. 使用 all_offload 求解器求解
    final_objective, solution, total_time = solve_all_offload(client_params, helper_params, network_params)
    
    # 提取能耗和效用
    energies = solution.get('energies', {})
    utilities = solution.get('utilities', {})
    
    # 计算总和
    total_energy = sum(energies.values())
    total_utility = sum(utilities.values())

    # 3. 准备日志记录
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': seed,
        'num_clients': num_clients,
        'num_helpers': num_helpers,
        'status': 'All_Offload',  # 标记为基准算法
        'final_lb': final_objective, # 对于确定性算法，上下界相同
        'final_ub': final_objective,
        'total_time': total_time,
        'solution_x': str(solution['x_i']),
        'total_energy': total_energy,
        'total_utility': total_utility
    }
    
    # 4. 写入日志
    fieldnames = [
        'timestamp', 'seed', 'num_clients', 'num_helpers', 'status',
        'final_lb', 'final_ub', 'total_time', 'solution_x',
        'total_energy', 'total_utility'
    ]
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(log_entry)
        
    print(f"--- Finished. Objective={final_objective:.4f}, Time={total_time:.4f}s ---")

if __name__ == '__main__':
    setup_logger()
    
    for n_clients in CLIENT_SCALES:
        print(f"\n=======================================================")
        print(f"Starting experiments for {n_clients} clients...")
        print(f"=======================================================")
        
        # 每个规模运行数次，使用真正的随机种子
        for i in range(SEEDS_PER_SCALE):
            # 生成一个范围较大的随机整数作为种子
            seed = random.randint(1, 1_000_000_000)
            run_single_experiment(num_clients=n_clients, num_helpers=NUM_HELPERS, seed=seed)
            
    print("\nAll baseline experiments completed.")