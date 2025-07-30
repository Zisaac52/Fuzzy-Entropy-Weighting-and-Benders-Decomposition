# main.py

import sys
import os
import random
import logging
import csv # 新增
import datetime # 新增
import time # 新增
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pe

# --- 1. 设置系统路径和日志 ---
# 确保Python可以找到我们的包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# --- 2. 导入项目模块 ---
# 使用绝对路径导入，更稳健
from gbd_resource_optimizer.gbd_core import master_problem, sub_problem, cuts
from gbd_resource_optimizer import problem_def
from gbd_resource_optimizer.data_utils import instance_generator



#--- 3. 固定随机种子以保证实验可复现 ---
RANDOM_SEED = 1179084202
# random.seed(RANDOM_SEED) # Commented out: Seed will be set at the start of run_gbd

RESULTS_CSV_FILE = "gbd_run_summary_log.csv" # 定义全局变量

def run_gbd(num_clients_cfg, num_helpers_cfg, heterogeneity_level_cfg):
    """
    执行广义Benders分解算法的主函数。
    Args:
        num_clients_cfg (int): 客户端数量配置.
        num_helpers_cfg (int): 辅助节点数量配置.
        heterogeneity_level_cfg (str): 异构性级别配置.
    """
    start_time = time.time() # 记录开始时间

    # 确定并设置本次运行的种子
    seed_to_use = None
    if 'RANDOM_SEED' in globals() and isinstance(globals()['RANDOM_SEED'], int):
        seed_to_use = globals()['RANDOM_SEED']
        logging.info(f"Using globally/externally set RANDOM_SEED: {seed_to_use}")
    else:
        # RANDOM_SEED 未被有效设置（可能被注释，或不是整数），则生成一个新的随机种子
        seed_to_use = random.randint(0, 2**32 - 1)
        logging.info(f"Global RANDOM_SEED not set or invalid. Generated a new random seed for this run: {seed_to_use}")
    
    random.seed(seed_to_use)
    
    # 将实际使用的种子存储在一个局部变量，以便最后打印，
    # 避免在函数执行过程中全局 RANDOM_SEED 可能被意外修改（尽管不太可能）
    effective_seed_for_this_run = seed_to_use

    # =========================================================================
    # 步骤 A: 初始化
    # =========================================================================
    print("=====================================================")
    print("GBD Resource Optimizer for Federated Learning - START")
    print("=====================================================\n")

    # --- GBD 控制参数 ---
    max_iterations = 100
    epsilon_convergence = 1e-6
    lower_bound = -float('inf')
    upper_bound = float('inf')
    iteration = 0
    
    # 用于存储最佳解的变量
    best_solution_int = None
    best_solution_cont = None
    best_upper_bound = float('inf')

    # --- 实验规模配置 (从参数传入) ---
    num_clients = num_clients_cfg
    num_helpers = num_helpers_cfg
    heterogeneity_level = heterogeneity_level_cfg


    # --- 生成问题实例 ---
    client_params_list, helper_params_list, network_params = \
        instance_generator.generate_instance(num_clients=num_clients, num_helpers=num_helpers, heterogeneity_level=heterogeneity_level)

    logging.info(f"Generated instance with {num_clients} clients, {num_helpers} helpers, heterogeneity: {heterogeneity_level}.")

    # =========================================================================
    # 步骤 B: 构建模型和初始解
    # =========================================================================
    
    # --- 构建主问题模型 (只构建一次) ---
    master_model = master_problem.build_master_problem_model(
        client_params_list,
        helper_params_list,
        network_params
    )

    # --- 为第一次迭代手动创建初始整数解 ---
    # 策略：所有客户端都进行本地计算 (x_i=1)，N_i取一个中间值，Y_ij都为0
    initial_fixed_int_vars = {'x_i': {}, 'N_i': {}, 'Y_ij': {}}
    for cp in client_params_list:
        initial_fixed_int_vars['x_i'][cp.id] = 1
        initial_fixed_int_vars['N_i'][cp.id] = int(network_params.N_f / 2) # 或设为1
        for hp in helper_params_list:
            initial_fixed_int_vars['Y_ij'][(cp.id, hp.id)] = 0
    
    # current_int_solution 用于在迭代中传递整数解
    current_int_solution = initial_fixed_int_vars

    # =========================================================================
    # 步骤 C: GBD 迭代主循环 (SP -> MP -> SP ...)
    # =========================================================================
    print("\nGBD Algorithm Started...")
    print("----------------------------------------------------")

    while iteration < max_iterations and upper_bound - lower_bound > epsilon_convergence:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # --- 1. 求解子问题 (SP) ---
        print("Building and Solving Subproblem with current integer solution...")
        sub_model = sub_problem.build_sub_problem_model(
            current_int_solution, 
            client_params_list, 
            helper_params_list, 
            network_params
        )
        sp_status, sp_term_cond, sp_obj_val, sp_cont_sol, sp_duals = sub_problem.solve_sub_problem(sub_model, tee=False)
        
        # --- 2. 检查SP状态，更新上界(UB)，生成割 ---
        if sp_status == SolverStatus.ok and (sp_term_cond == TerminationCondition.optimal or sp_term_cond == TerminationCondition.locallyOptimal):
            # 子问题求解成功且可行
            
            # 更新上界 UB (对于min问题，UB是历史最小)
            if sp_obj_val < upper_bound:
                upper_bound = sp_obj_val
                best_solution_int = current_int_solution.copy()
                best_solution_cont = sp_cont_sol.copy() if sp_cont_sol else {}
                print(f"New best solution found. UB updated to: {upper_bound:.4f}")
            else:
                print(f"SP Solved (Feasible). Objective: {sp_obj_val:.4f}. UB not improved (current UB: {upper_bound:.4f})")

            # 生成并添加最优性割
            print("Generating optimality cut...")
            optimality_cut = cuts.generate_optimality_cut_expression(
                current_int_solution, sp_obj_val, sp_duals, sp_cont_sol,
                master_model, client_params_list, helper_params_list, network_params
            )
            master_problem.add_benders_cut_to_master(master_model, optimality_cut)
            print("Optimality cut added to Master Problem.")

        elif sp_term_cond == TerminationCondition.infeasible:
            # 子问题不可行，需要生成可行性割
            print("SP Infeasible. Building and Solving Relaxed Subproblem...")
            relaxed_sub_model = sub_problem.build_relaxed_sub_problem_model(
                current_int_solution, client_params_list, helper_params_list, network_params
            )
            # **确保 solve_relaxed_sub_problem 返回连续解**
            rsp_status, rsp_term_cond, relaxed_sp_obj_val, relaxed_sp_cont_sol, relaxed_sp_duals = \
                sub_problem.solve_relaxed_sub_problem(relaxed_sub_model, tee=False)

            if rsp_status == SolverStatus.ok and (rsp_term_cond == TerminationCondition.optimal or rsp_term_cond == TerminationCondition.locallyOptimal):
                if relaxed_sp_obj_val > 1e-6: # 检查松弛变量之和是否大于0
                    print(f"Relaxed SP Solved. Objective (sum of violations): {relaxed_sp_obj_val:.4f}. Generating feasibility cut.")
                    feasibility_cut = cuts.generate_feasibility_cut_expression(
                        current_int_solution,
                        relaxed_sp_obj_val,
                        relaxed_sp_duals,
                        relaxed_sp_cont_sol, # <-- 传递松弛子问题的连续解
                        master_model,
                        client_params_list,
                        helper_params_list,
                        network_params
                    )
                    master_problem.add_benders_cut_to_master(master_model, feasibility_cut)
                    print("Feasibility cut added to Master Problem.")
                else:
                    print("Warning: Relaxed SP objective is close to zero, but original SP was infeasible. Check problem or tolerances. No cut added.")
            else:
                print(f"ERROR: Relaxed subproblem could not be solved. Cannot generate feasibility cut. Stopping GBD.")
                print(f"  Solver Status: {rsp_status}, Termination Condition: {rsp_term_cond}")
                break
        else:
            print(f"ERROR: Subproblem solver failed or returned unexpected status. Stopping GBD.")
            print(f"  Solver Status: {sp_status}, Termination Condition: {sp_term_cond}")
            break

        # --- 3. 求解主问题 (MP) ---
        print("Solving Master Problem...")
        mp_obj_val, next_int_solution, mp_status, mp_term_cond = master_problem.solve_master_problem(master_model, tee=False)
 
        if not (mp_status == SolverStatus.ok and mp_term_cond == TerminationCondition.optimal):
            print(f"ERROR: Master problem could not be solved to optimality. Stopping.")
            print(f"  Solver Status: {mp_status}, Termination Condition: {mp_term_cond}")
            break
        
        # 更新下界 LB
        lower_bound = mp_obj_val
        print(f"MP Solved. Objective (Lower Bound): {lower_bound:.4f}")
        
        # --- 4. 更新整数解用于下一次迭代 ---
        if next_int_solution and any(next_int_solution.values()):
             current_int_solution = next_int_solution
        else:
             print("ERROR: Master problem returned an empty or invalid solution. GBD cannot proceed.")
             break

        # --- 5. 打印迭代信息 ---
        print(f"End of Iteration {iteration}: LB = {lower_bound:.4f}, UB = {upper_bound:.4f}")
        if upper_bound != float('inf') and lower_bound != -float('inf'):
            gap = upper_bound - lower_bound
            relative_gap_str = f"({100 * gap / (abs(upper_bound) + 1e-9):.2f}%)" if abs(upper_bound) > 1e-9 else ""
            print(f"Current Gap: {gap:.4f} {relative_gap_str}")
        else:
            print("Current Gap: inf")
        print("----------------------------------------------------")

    # =========================================================================
    # 步骤 D: 最终结果输出
    # =========================================================================
    print("\nGBD Algorithm Finished.")
    print("====================================================")
    
    final_gap = upper_bound - lower_bound if upper_bound != float('inf') and lower_bound != -float('inf') else float('inf')
    solution_status = "Unknown"
    if upper_bound != float('inf') and final_gap <= epsilon_convergence:
        solution_status = "Converged"
        print(f"Converged successfully in {iteration} iterations.")
    elif iteration >= max_iterations:
        solution_status = "Max Iterations Reached"
        print(f"Reached maximum iterations ({max_iterations}) without converging.")
    elif best_solution_int is None and upper_bound == float('inf'): # 检查是否早期终止且无解
        solution_status = "No Feasible Solution Found / Terminated Early"
        print("Algorithm terminated prematurely. No feasible solution was found.")
    else: # 其他提前终止的情况 (可能MP或SP出错)
        solution_status = "Terminated Prematurely"
        print("Algorithm terminated prematurely.")
        
    total_gbd_time = time.time() - start_time
    print(f"Total GBD Time (s): {total_gbd_time:.2f}")
    print(f"Final Lower Bound (LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Best UB Found): {f'{best_upper_bound:.4f}' if best_upper_bound != float('inf') else 'Infinity'}")
    print(f"Final Gap: {f'{final_gap:.4f}' if final_gap != float('inf') else 'Infinity'}")
    print(f"Total Iterations: {iteration}")

    if best_solution_int:
        print("\nBest Feasible Solution Found:")
        print(f"  Objective Value (Best UB): {best_upper_bound:.4f}")
        print(f"  Integer Variables (x_i): {best_solution_int.get('x_i', {})}")
        print(f"  Integer Variables (N_i): {best_solution_int.get('N_i', {})}")
        print(f"  Integer Variables (Y_ij): {best_solution_int.get('Y_ij', {})}")
        # print(f"  Continuous Variables (from SP that yielded best UB): {best_solution_cont}") # 可能过长
    elif solution_status != "Converged": # 避免在未找到解时重复打印
        print("\nNo feasible integer solution was found or recorded as best.")
    print("====================================================")
    
    seed_for_log = effective_seed_for_this_run
    logging.info(f"Run completed. Effective RANDOM_SEED for this run: {seed_for_log}")

    results_summary = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Instance_Name": f"c{num_clients}_h{num_helpers}_s{seed_for_log}_het{heterogeneity_level[:1]}",
        "Num_Clients": num_clients,
        "Num_Helpers": num_helpers,
        "Heterogeneity_Level": heterogeneity_level,
        "Final_Objective_Value": best_upper_bound if best_upper_bound != float('inf') else 'inf',
        "Total_GBD_Time_s": round(total_gbd_time, 2),
        "Total_Iterations": iteration,
        "Final_LB": lower_bound if lower_bound != -float('inf') else '-inf',
        "Final_UB": best_upper_bound if best_upper_bound != float('inf') else 'inf',
        "Final_Gap": final_gap if final_gap != float('inf') else 'inf',
        "Solution_Status": solution_status,
        "Optimal_x_i": str(best_solution_int.get('x_i', {})) if best_solution_int else "N/A",
        "Optimal_N_i": str(best_solution_int.get('N_i', {})) if best_solution_int else "N/A",
        "Optimal_Y_ij": str(best_solution_int.get('Y_ij', {})) if best_solution_int else "N/A",
        "Optimal_Continuous_Solution": str(best_solution_cont) if best_solution_cont else "N/A" # 可考虑截断或选择部分
    }
    return results_summary

def log_results_to_csv(results_dict, csv_filepath):
    """将单次运行的结果字典写入CSV文件"""
    if not results_dict:
        return

    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = results_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)
    logging.info(f"Results for instance {results_dict.get('Instance_Name')} logged to {csv_filepath}")


if __name__ == "__main__":
    # 定义将要传递给 run_gbd 的参数
    # 这些值可以来自配置文件、命令行参数，或者像这样硬编码用于单次测试
    num_clients_run = 2 # 和文件顶部的默认值一致
    num_helpers_run = 1  # 和文件顶部的默认值一致
    heterogeneity_level_run = 'medium' # 和文件顶部的默认值一致
    
    # 如果 RANDOM_SEED 没有被外部脚本修改，则使用文件顶部的定义
    # 如果 RANDOM_SEED 被注释或未定义，run_gbd内部会生成一个
    current_seed = globals().get('RANDOM_SEED') # 获取在 run_gbd 之外定义的 RANDOM_SEED

    # 运行GBD算法
    summary_data = run_gbd(num_clients_run, num_helpers_run, heterogeneity_level_run)
    
    # 记录结果到CSV
    log_results_to_csv(summary_data, RESULTS_CSV_FILE)