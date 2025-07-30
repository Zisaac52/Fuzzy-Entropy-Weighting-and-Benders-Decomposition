import os
import csv
import random
import datetime
import logging
from gbd_resource_optimizer import main as gbd_main

# --- 配置参数 ---
SEEDS_PER_CONFIG = 500  # 每个配置要测试的种子数量
MIN_RANDOM_SEED = 0
MAX_RANDOM_SEED = 2**32 - 1
OUTPUT_CSV_FILE = "comprehensive_gbd_runs_log.csv"
LOG_FILE = "find_seeds.log"
EPSILON_CONVERGENCE = 1e-6 # GBD收敛容差

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()])

# --- 运行状态定义 ---
STATUS_CONVERGED = 'Converged'
STATUS_INFEASIBLE = 'Infeasible'
STATUS_NOT_CONVERGED = 'Not Converged'
STATUS_EXCEPTION = 'Exception'

# --- 实验配置 ---
# 调整实验配置，专注于更可能产生非平凡解（有客户端卸载）的场景
# 主要通过降低 strong_client_ratio 实现，让更多“弱客户端”出现
# 弱客户端算力低、预算紧张，更有可能需要卸载
search_configs = [
    # # C=2, H=1, 弱客户端为主
    # {'clients': 2, 'helpers': 1, "strong_ratio": 0.0},
    # # C=5, H=1, 逐步增加强客户端比例
    # {'clients': 5, 'helpers': 1, "strong_ratio": 0.2},
    # {'clients': 5, 'helpers': 1, "strong_ratio": 0.4},
    # # C=8, H=1, 弱客户端为主
    # {'clients': 8, 'helpers': 1, "strong_ratio": 0.25},
    # C=10, H=2, 弱客户端为主
    {'clients': 10, 'helpers': 2, "strong_ratio": 0.8},
    # C=15, H=2, 弱客户端为主
    {'clients': 15, 'helpers': 2, "strong_ratio": 0.8},
    {'clients': 20, 'helpers': 2, "strong_ratio": 0.8},
    {'clients': 25, 'helpers': 2, "strong_ratio": 0.8},
    {'clients': 30, 'helpers': 2, "strong_ratio": 0.8},
]

def log_run_to_csv(timestamp, seed, num_clients, num_helpers, status, final_lb, final_ub, total_time, solution_x):
    """将单次运行的结果追加到CSV文件。"""
    with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'seed', 'num_clients', 'num_helpers', 'status', 'final_lb', 'final_ub', 'total_time', 'solution_x']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'timestamp': timestamp,
            'seed': seed,
            'num_clients': num_clients,
            'num_helpers': num_helpers,
            'status': status,
            'final_lb': final_lb,
            'final_ub': final_ub,
            'total_time': f"{total_time:.4f}",
            'solution_x': str(solution_x)
        })

# --- 主逻辑 ---
if __name__ == "__main__":
    logging.info("Starting comprehensive GBD testing to log all outcomes.")

    # 在实验开始前，初始化/清空CSV文件并写入表头
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'seed', 'num_clients', 'num_helpers', 'status', 'final_lb', 'final_ub', 'total_time', 'solution_x']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for config_idx, config in enumerate(search_configs):
        num_c = config["clients"]
        num_h = config["helpers"]
        strong_r = config["strong_ratio"]

        logging.info(f"===== Testing Config {config_idx+1}/{len(search_configs)}: {num_c} Clients, {num_h} Helpers, Strong Ratio: {strong_r} =====")

        for i in range(SEEDS_PER_CONFIG):
            seed = random.randint(MIN_RANDOM_SEED, MAX_RANDOM_SEED)
            status = ''
            lb, ub = -1, -1
            
            start_time = datetime.datetime.now()
            
            try:
                # 直接调用GBD核心函数，并接收整数解
                lb, ub, solution_int, _, used_seed = gbd_main.run_gbd(
                    num_clients=num_c,
                    num_helpers=num_h,
                    seed=seed,
                    strong_client_ratio=strong_r
                )

                # 判断最终状态
                if (ub - lb) <= EPSILON_CONVERGENCE:
                    status = STATUS_CONVERGED
                elif ub == float('inf'):
                    status = STATUS_INFEASIBLE
                else:
                    status = STATUS_NOT_CONVERGED

            except Exception as e:
                logging.error(f"An exception occurred with seed {seed}: {e}", exc_info=False) # exc_info=False to avoid flooding logs
                status = STATUS_EXCEPTION
                solution_int = None # 确保异常时 solution_int 为 None
                # 即使有异常，也要确保used_seed有值用于记录
                used_seed = seed
            
            end_time = datetime.datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # 仅在找到可行整数解时记录到CSV
            # 仅在找到至少有一个客户端卸载（x_i=1）的可行解时记录
            if solution_int and 'x_i' in solution_int and any(solution_int['x_i'].values()):
                log_run_to_csv(
                    timestamp=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    seed=used_seed,
                    num_clients=num_c,
                    num_helpers=num_h,
                    status=status,
                    final_lb=lb,
                    final_ub=ub,
                    total_time=total_time,
                    solution_x=solution_int['x_i']
                )

            logging.info(f"Finished seed {i+1}/{SEEDS_PER_CONFIG} for config {config_idx+1}. Status: {status}, Time: {total_time:.2f}s")
            
    logging.info("Experiment finished. All seeds have been tested.")