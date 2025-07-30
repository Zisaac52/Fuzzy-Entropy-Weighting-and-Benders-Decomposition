# run_experiment.py

import argparse
import sys
import os
import random
import logging
import time
import csv

# --- 将项目根目录添加到Python路径中 ---
# 这使得我们可以轻松地从项目根目录导入模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- 导入项目模块 ---
from gbd_resource_optimizer.main import run_gbd
from gbd_resource_optimizer.baselines.local_first import solve_local_first
from gbd_resource_optimizer.baselines.greedy_offload import solve_greedy_offload
from gbd_resource_optimizer.baselines.random_offload import solve_random_offload
from gbd_resource_optimizer.baselines.all_offload import solve_all_offload
from gbd_resource_optimizer.data_utils import instance_generator

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gbd_optimizer.log',
    filemode='a' 
)

def log_to_comprehensive_csv(summary_data, log_file):
    """
    将单次运行的总结以一行的形式，追加到主CSV日志文件中。
    如果文件不存在，则创建并写入表头。
    """
    
    # --- 定义表头 ---
    # 确保这里的键与 summary_log 中的键一致且顺序正确
    header = [
        'timestamp', 'run_id', 'solver', 'num_clients', 'num_helpers', 'seed',
        'strong_ratio', 'task_workload', 'status', 'total_time', 'total_mp_time',
        'total_sp_time', 'total_rsp_time', 'total_iterations', 'final_objective_ub',
        'final_lower_bound_lb', 'solution_x_i', 'final_energies', 'final_utilities',
        'total_energy', 'total_utility'
    ]

    # --- 准备要写入的数据行 ---
    # 使用 get(key, None) 来安全地获取数据，如果键不存在则为 None
    row_data = {key: summary_data.get(key) for key in header}
    row_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            
            if not file_exists:
                writer.writeheader()  # 如果文件是新创建的，则写入表头
            
            writer.writerow(row_data)
        
        print(f"Successfully appended run results to: {log_file}")

    except IOError as e:
        print(f"Error writing to comprehensive CSV file {log_file}: {e}")
        logging.error(f"Failed to write to {log_file}: {e}")


def main():
    """
    实验执行的主函数
    """
    parser = argparse.ArgumentParser(description="Run GBD and baseline experiments for resource optimization.")
    
    parser.add_argument(
        '--solver',
        type=str,
        default='gbd',
        choices=['gbd', 'local_first', 'greedy_offload', 'random_offload', 'all_offload', 'all'],
        help="The solver to use. 'all' will run GBD and all baseline algorithms."
    )
    parser.add_argument('--num_clients', type=int, default=5, help="Number of clients in the scenario.")
    parser.add_argument('--num_helpers', type=int, default=1, help="Number of helpers in the scenario.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--strong_ratio', type=float, default=0.5, help="Ratio of strong clients in the instance.")
    parser.add_argument('--log_file', type=str, default='gbd_main_experiment_log.csv', help="Name of the log file.")
    parser.add_argument('--task_workload', type=int, default=None, help="Fixed task workload in KB for all clients.")
    parser.add_argument('--run_id', type=str, default=None, help="A unique identifier for a single experimental run.")
    
    args = parser.parse_args()
    
    # --- 如果没有提供种子，就生成一个 ---
    if args.seed is None:
        seed_to_use = random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using generated seed: {seed_to_use}")
    else:
        seed_to_use = args.seed
        print(f"Using provided seed: {seed_to_use}")
        
    random.seed(seed_to_use)
    logging.info(f"Experiment started with args: {args}")
    logging.info(f"Effective seed for this run: {seed_to_use}")

    # --- 1. 生成所有算法共享的问题实例 ---
    print("\nGenerating problem instance...")
    client_params_list, helper_params_list, network_params = \
        instance_generator.generate_instance(
            num_clients=args.num_clients,
            num_helpers=args.num_helpers,
            strong_client_ratio=args.strong_ratio,
            task_workload=args.task_workload
        )
    print(f"Instance generated for {args.num_clients} clients and {args.num_helpers} helpers.")

    # 将生成的实例打包
    instance = (client_params_list, helper_params_list, network_params)

    # --- 2. 根据solver参数调用相应函数 ---
    solvers_to_run = []
    if args.solver == 'all':
        solvers_to_run = ['gbd', 'local_first', 'greedy_offload', 'random_offload', 'all_offload']
    else:
        solvers_to_run = [args.solver]
        
    results_summary_for_print = {}
    gbd_status = None  # 用于跟踪 GBD 的最终状态

    for solver_name in solvers_to_run:
        print(f"\n==================== RUNNING SOLVER: {solver_name.upper()} ====================")
        
        summary_log = {
            "solver": solver_name,
            "num_clients": args.num_clients,
            "num_helpers": args.num_helpers,
            "strong_ratio": args.strong_ratio,
            "seed": seed_to_use,
            "task_workload": args.task_workload,
            "run_id": args.run_id
        }

        if solver_name == 'gbd':
            gbd_results = run_gbd(
                seed=seed_to_use,
                instance=instance
            )
            # 更新日志字典
            summary_log.update(gbd_results)
            gbd_status = gbd_results.get('status')  # 获取 GBD 运行状态

            # 提取 x_i 的解
            if gbd_results.get('best_solution_int') and 'x_i' in gbd_results['best_solution_int']:
                summary_log['solution_x_i'] = str(gbd_results['best_solution_int']['x_i'])
                final_energies = gbd_results.get('final_energies', {})
                final_utilities = gbd_results.get('final_utilities', {})
                summary_log['final_energies'] = str(final_energies)
                summary_log['final_utilities'] = str(final_utilities)
                summary_log['total_energy'] = sum(final_energies.values())
                summary_log['total_utility'] = sum(final_utilities.values())

            # 为最终打印准备数据
            results_summary_for_print['gbd'] = {'Objective': gbd_results.get('final_objective_ub')}

        elif solver_name == 'local_first':
            start_time = time.time()
            cost, solution, makespan = solve_local_first(
                client_params_list, helper_params_list, network_params
            )
            energies = solution.get('energies', {})
            utilities = solution.get('utilities', {})
            summary_log.update({
                "status": "Optimal", "total_time": makespan, "final_objective_ub": cost,
                "solution_x_i": str(solution.get('x_i')) if solution else None,
                "total_energy": sum(energies.values()),
                "total_utility": sum(utilities.values())
            })
            results_summary_for_print['local_first'] = {'Objective': cost}

        elif solver_name == 'greedy_offload':
            start_time = time.time()
            cost, solution, makespan = solve_greedy_offload(
                client_params_list, helper_params_list, network_params
            )
            energies = solution.get('energies', {})
            utilities = solution.get('utilities', {})
            summary_log.update({
                "status": "Heuristic", "total_time": makespan, "final_objective_ub": cost,
                "solution_x_i": str(solution.get('x_i')) if solution else None,
                "total_energy": sum(energies.values()),
                "total_utility": sum(utilities.values())
            })
            results_summary_for_print['greedy_offload'] = {'Objective': cost}

        elif solver_name == 'random_offload':
            start_time = time.time()
            cost, solution, makespan = solve_random_offload(
                client_params_list, helper_params_list, network_params
            )
            energies = solution.get('energies', {})
            utilities = solution.get('utilities', {})
            summary_log.update({
                "status": "Heuristic", "total_time": makespan, "final_objective_ub": cost,
                "solution_x_i": str(solution.get('x_i')) if solution else None,
                "total_energy": sum(energies.values()),
                "total_utility": sum(utilities.values())
            })
            results_summary_for_print['random_offload'] = {'Objective': cost}

        elif solver_name == 'all_offload':
            start_time = time.time()
            cost, solution, makespan = solve_all_offload(
                client_params_list, helper_params_list, network_params
            )
            energies = solution.get('energies', {})
            utilities = solution.get('utilities', {})
            summary_log.update({
                "status": "Heuristic", "total_time": makespan, "final_objective_ub": cost,
                "solution_x_i": str(solution.get('x_i')) if solution else None,
                "total_energy": sum(energies.values()),
                "total_utility": sum(utilities.values())
            })
            results_summary_for_print['all_offload'] = {'Objective': cost}
            
        # --- 每次运行后调用日志函数 ---
        log_to_comprehensive_csv(summary_log, args.log_file)
            
        print(f"==================== FINISHED SOLVER: {solver_name.upper()} ====================")
        
    # --- 3. 打印最终总结 ---
    print("\n\n==================== EXPERIMENT SUMMARY ====================")
    print(f"Configuration: {args.num_clients} Clients, {args.num_helpers} Helpers, Seed: {seed_to_use}")
    print("------------------------------------------------------------")
    for solver, result in results_summary_for_print.items():
        objective_val = result.get('Objective', 'N/A')
        obj_str = f"{objective_val:.4f}" if isinstance(objective_val, (int, float)) and objective_val != float('inf') else "Infinity"
        print(f"Solver: {solver.ljust(15)} | Final Objective: {obj_str}")
    print("============================================================")


    # --- 4. 根据 GBD 状态码退出 ---
    # 如果 gbd_status 被设置了（意味着 GBD 求解器被运行了）
    if gbd_status:
        if gbd_status == 'Converged':
            print(f"GBD Converged. Exiting with status 0.")
            logging.info("GBD Converged. Exiting with status 0.")
            sys.exit(0)
        else:
            print(f"GBD did not converge (Status: {gbd_status}). Exiting with status 1.")
            logging.warning(f"GBD did not converge (Status: {gbd_status}). Exiting with status 1.")
            sys.exit(1)


if __name__ == '__main__':
    main()
