# main.py

import sys
import os
import random
import logging
import time # <--- 添加导入 time 模块
import csv
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pe

# --- 1. 导入项目模块 ---
# 使用绝对路径导入，更稳健
from gbd_resource_optimizer.gbd_core import master_problem, sub_problem, cuts
from gbd_resource_optimizer import problem_def
from gbd_resource_optimizer.data_utils import instance_generator



#--- 3. 固定随机种子以保证实验可复现 ---
# RANDOM_SEED = 1558738043
# random.seed(RANDOM_SEED) # Commented out: Seed will be set at the start of run_gbd

def run_gbd(num_clients=None, num_helpers=None, seed=None, strong_client_ratio=0.5, task_workload=None,
            instance=None):
    """
    执行广义Benders分解算法的主函数。
    可以接收一个预先生成的实例，或者根据参数生成新实例。
    """
    start_time = time.time() # <--- 记录开始时间

    # 确定并设置本次运行的种子
    seed_to_use = None
    if seed is None:
        # 如果没有提供种子，则生成一个随机种子
        seed_to_use = random.randint(0, 2**32 - 1)
        logging.info(f"No seed provided. Generated a new random seed for this run: {seed_to_use}")
    else:
        # 使用提供的种子
        seed_to_use = seed
        logging.info(f"Using provided seed for this run: {seed_to_use}")
    
    random.seed(seed_to_use)
    
    # 将实际使用的种子存储在一个局部变量，以便最后打印，
    # 避免在函数执行过程中全局 RANDOM_SEED 可能被意外修改（尽管不太可能）
    effective_seed_for_this_run = seed_to_use

    # =========================================================================
    # 步骤 A: 初始化
    # =========================================================================
    
    # --- 日志文件设置 ---
    csv_log_file = 'gbd_iteration_details.csv'
    csv_header = [
        'iteration', 'lower_bound', 'upper_bound', 'mp_time', 'sp_time',
        'rsp_time', 'total_mp_time', 'total_sp_time', 'total_rsp_time', 'gap'
    ]
    with open(csv_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print("=====================================================")
    print("GBD Resource Optimizer for Federated Learning - START")
    print("=====================================================\n")

    # --- GBD 控制参数 ---
    max_iterations = 10
    epsilon_convergence = 1e-6
    lower_bound = -float('inf')
    upper_bound = float('inf')
    iteration = 0
    
    # 计时和计数器初始化
    total_mp_time = 0.0
    mp_solve_count = 0
    total_sp_time = 0.0
    sp_solve_count = 0
    total_rsp_time = 0.0
    rsp_solve_count = 0
    
    # 用于存储最佳解的变量
    best_solution_int = None
    best_solution_cont = None
    best_upper_bound = float('inf')

    # --- 实验规模配置 (可调整) ---

    # --- 问题实例处理 ---
    if instance:
        client_params_list, helper_params_list, network_params = instance
        num_clients = len(client_params_list)
        num_helpers = len(helper_params_list)
        logging.info(f"Using pre-generated instance with {num_clients} clients and {num_helpers} helpers.")
    else:
        if num_clients is None or num_helpers is None:
            raise ValueError("Must provide num_clients and num_helpers if instance is not given.")
        logging.info(f"Generating new instance with {num_clients} clients and {num_helpers} helpers.")
        client_params_list, helper_params_list, network_params = \
            instance_generator.generate_instance(
                num_clients=num_clients,
                num_helpers=num_helpers,
                heterogeneity_level='medium',
                strong_client_ratio=strong_client_ratio,
                task_workload=task_workload
            )

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
        mp_time, sp_time, rsp_time = 0.0, 0.0, 0.0
        # 在每次迭代开始时，重置所有单次迭代的计时器
        print(f"\n--- Iteration {iteration} ---")

        # --- 1. 求解子问题 (SP) ---
        print("Building and Solving Subproblem with current integer solution...")
        print(f"  [DEBUG] Current Integer Solution for SP: {current_int_solution}") # <--- 增加日志
        sub_model = sub_problem.build_sub_problem_model(
            current_int_solution,
            client_params_list,
            helper_params_list,
            network_params
        )
        sp_time_start = time.time()
        # v-- 开启求解器日志
        sp_status, sp_term_cond, sp_obj_val, sp_cont_sol, sp_duals = sub_problem.solve_sub_problem(sub_model, tee=True)
        sp_time = time.time() - sp_time_start
        print(f"  [DEBUG] SP Termination Condition: {sp_term_cond}") # <--- 增加日志
        total_sp_time += sp_time
        sp_solve_count += 1
        
        # --- 2. 检查SP状态，更新上界(UB)，生成割 ---
        if sp_status == SolverStatus.ok and (sp_term_cond == TerminationCondition.optimal or sp_term_cond == TerminationCondition.locallyOptimal):
            # 子问题求解成功且可行
            
            # 更新上界 UB (对于min问题，UB是历史最小)
            if sp_obj_val < upper_bound:
                upper_bound = sp_obj_val
                best_upper_bound = sp_obj_val
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
            rsp_time_start = time.time()
            # v-- 开启求解器日志
            rsp_status, rsp_term_cond, relaxed_sp_obj_val, relaxed_sp_cont_sol, relaxed_sp_duals = \
                sub_problem.solve_relaxed_sub_problem(relaxed_sub_model, tee=True)
            rsp_time = time.time() - rsp_time_start
            total_rsp_time += rsp_time
            rsp_solve_count += 1

            if rsp_status == SolverStatus.ok and (rsp_term_cond == TerminationCondition.optimal or rsp_term_cond == TerminationCondition.locallyOptimal):
                if relaxed_sp_obj_val > 1e-6: # 检查松弛变量之和是否大于0
                    print(f"Relaxed SP Solved. Objective (sum of violations): {relaxed_sp_obj_val:.4f}. Generating feasibility cut.")
                    
                    # <--- 增加日志：打印所有值大于零的松弛变量
                    print("  [DEBUG] Relaxed SP Slack Variables (s > 0):")
                    if relaxed_sp_cont_sol:
                        for var_name, values in relaxed_sp_cont_sol.items():
                             # 松弛变量通常以's_'开头
                            if var_name.startswith('s_') and isinstance(values, dict):
                                for key, val in values.items():
                                    if val > 1e-6:
                                        print(f"    {var_name}[{key}] = {val:.6f}")

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
                    # <--- 增加日志：打印生成的可行性割
                    print(f"  [DEBUG] Generated Feasibility Cut Expression:\n    {feasibility_cut}")
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
        mp_time_start = time.time()
        # v-- 开启求解器日志
        mp_obj_val, next_int_solution, mp_status, mp_term_cond = master_problem.solve_master_problem(master_model, tee=True)
        mp_time = time.time() - mp_time_start
        print(f"  [DEBUG] MP Termination Condition: {mp_term_cond}") # <--- 增加日志
        total_mp_time += mp_time
        mp_solve_count += 1
 
        if not (mp_status == SolverStatus.ok and mp_term_cond == TerminationCondition.optimal):
            if mp_term_cond == TerminationCondition.infeasible:
                print("Master problem is infeasible. This implies the optimal solution has been found and the algorithm will now terminate.")
                # 仅跳出循环，在函数末尾统一处理返回值
            else:
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
            gap = float('inf')
            print("Current Gap: inf")
        
        # --- 6. 将迭代详情写入CSV ---
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row_data = [
                iteration, lower_bound, upper_bound, mp_time, sp_time, rsp_time,
                total_mp_time, total_sp_time, total_rsp_time, gap
            ]
            writer.writerow(row_data)

        print("----------------------------------------------------")

    # =========================================================================
    # 步骤 D: 最终结果输出
    # =========================================================================
    print("\nGBD Algorithm Finished.")
    print("====================================================")
    
    final_gap = upper_bound - lower_bound
    # 重新定义收敛条件
    converged = (best_upper_bound != float('inf') and upper_bound - lower_bound <= epsilon_convergence)

    if converged:
        print(f"Converged successfully in {iteration} iterations.")
    elif mp_term_cond == TerminationCondition.infeasible:
        if best_upper_bound != float('inf'):
            print(f"Converged in {iteration} iterations by proving no other feasible solutions exist.")
        else:
            print(f"Converged in {iteration} iterations by proving the problem is INFEASIBLE.")
    elif iteration >= max_iterations:
        print(f"Reached maximum iterations ({max_iterations}) without converging.")
    else:
        print("Algorithm terminated prematurely.")

    # 根据终止条件，格式化最终的上下界报告
    final_lb_to_report = lower_bound if lower_bound != -float('inf') else "-inf"
    final_ub_to_report = f"{best_upper_bound:.4f}" if best_upper_bound != float('inf') else "inf"

    if mp_term_cond == TerminationCondition.infeasible and best_upper_bound != float('inf'):
        final_lb_to_report = f"{best_upper_bound:.4f}"

    print(f"Final Lower Bound (LB): {final_lb_to_report}")
    print(f"Final Upper Bound (Best UB Found): {final_ub_to_report}")
    print(f"Total Iterations: {iteration}")
    print(f"Master Problem: Solved {mp_solve_count} times, Total Time = {total_mp_time:.2f}s")
    print(f"Subproblem (SP): Solved {sp_solve_count} times, Total Time = {total_sp_time:.2f}s")
    if rsp_solve_count > 0:
        print(f"Relaxed Subproblem (RSP): Solved {rsp_solve_count} times, Total Time = {total_rsp_time:.2f}s")

    if best_solution_int:
        print("\nBest Feasible Solution Found:")
        print(f"  Objective Value: {best_upper_bound if best_upper_bound == float('inf') else f'{best_upper_bound:.4f}'}")
        print(f"  Integer Variables: {best_solution_int}")
        print(f"  Continuous Variables (from SP that yielded best UB):")
        if best_solution_cont:
            for var_name, values in best_solution_cont.items():
                print(f"    {var_name}: {values}")
        else:
            print("    None")
    else:
        print("\nNo feasible solution was found during the iterations.")
    
    total_gbd_time = time.time() - start_time # <--- 计算总时间
    print(f"Total GBD Time (s): {total_gbd_time:.2f}") # <--- 打印总时间
    print(f"Effective RANDOM_SEED for this run: {effective_seed_for_this_run}")
    print("====================================================")
    
    # Log the seed value that was actually used for this run
    logging.info(f"Run completed. Effective RANDOM_SEED for this run: {effective_seed_for_this_run}")

    # --- 5. 结果封装 ---
    # 根据终止条件确定最终状态
    status = "Unknown"
    if converged:
        status = "Converged"
    elif mp_term_cond == TerminationCondition.infeasible:
        if best_upper_bound != float('inf'):
            status = "Converged (by proof)"
        else:
            status = "Infeasible"
    elif iteration >= max_iterations:
        status = "Max Iterations Reached"
    else:
        status = "Terminated Prematurely"


    final_lb = lower_bound
    # 如果通过主问题不可行证明了最优性，下界应等于上界
    if mp_term_cond == TerminationCondition.infeasible and best_upper_bound != float('inf'):
        final_lb = best_upper_bound

    # --- 6. (新增) 计算最终解的能耗和效用 ---
    final_energies, final_utilities = {}, {}
    if best_solution_int and best_solution_cont:
        final_energies, final_utilities = _calculate_final_metrics(
            best_solution_int, best_solution_cont,
            client_params_list, helper_params_list, network_params
        )

    results_summary = {
        "status": status,
        "total_time": total_gbd_time,
        "total_mp_time": total_mp_time,
        "total_sp_time": total_sp_time,
        "total_rsp_time": total_rsp_time,
        "total_iterations": iteration,
        "final_objective_ub": best_upper_bound,
        "final_lower_bound_lb": final_lb,
        "best_solution_int": best_solution_int,
        "best_solution_cont": best_solution_cont,
        "seed": effective_seed_for_this_run,
        "final_energies": final_energies,
        "final_utilities": final_utilities
    }
    
    logging.info(f"Run completed. Returning summary: {results_summary}")

    return results_summary

def _calculate_final_metrics(solution_int, solution_cont, client_params, helper_params, network_params):
    """根据最终解计算每个客户端的能耗和实际效用。"""
    energies = {}
    utilities = {}

    for cp in client_params:
        client_id = cp.id
        x_i = solution_int['x_i'].get(client_id)

        if x_i == 1:  # 本地计算
            f_B_val = solution_cont['f_B'].get(client_id, 0.0)
            t_b = problem_def.calculate_T_B_value(cp.D_i, cp.q_i_B, f_B_val)
            e_b = problem_def.calculate_E_B_value(cp.D_i, cp.q_i_B, f_B_val, cp.k_i)
            
            energies[client_id] = e_b
            utilities[client_id] = -t_b

        elif x_i == 0:  # 卸载
            f_L_val = solution_cont['f_L'].get(client_id, 0.0)
            p_val = solution_cont['p'].get(client_id, 0.0)
            
            # 计算能耗 E_r = E_encrypt + E_transmit
            e_encrypt = problem_def.calculate_E_encrypt_value(cp.D_i, cp.q_i_L, f_L_val, cp.k_i_encrypt)
            e_transmit = problem_def.calculate_E_transmit_value(cp.D_i, cp.a_i, p_val)
            energies[client_id] = e_encrypt + e_transmit

            # 计算效用 O_i = sum(Y_ij * U_ij)
            sum_Yij_Uij = 0.0
            n_i_val = solution_int['N_i'].get(client_id, 0)
            for hp in helper_params:
                y_ij = solution_int['Y_ij'].get((client_id, hp.id), 0)
                if y_ij == 1:
                    u_ij = problem_def.calculate_U_ij_value(
                        cp.a_i, p_val, n_i_val, network_params.N_f,
                        hp.S_j, hp.n_attacker_density_at_j
                    )
                    sum_Yij_Uij += u_ij
            utilities[client_id] = sum_Yij_Uij
    
    return energies, utilities

