# gbd_resource_optimizer/baselines/random_offload.py

import time
import random
from gbd_resource_optimizer.problem_def import (
    calculate_T_B_value, calculate_E_B_value,
    calculate_T_encrypt_value, calculate_T_transmit_value, calculate_T_compute_helper_value,
    calculate_E_encrypt_value, calculate_E_transmit_value, calculate_U_ij_value
)

def solve_random_offload(client_params_list, helper_params_list, network_params):
    """
    一个基准算法，为每个客户端随机选择一个执行策略（本地或卸载到随机辅助节点）。

    核心逻辑：
    - 遍历每一个客户端。
    - 随机决定是本地执行还是卸载。
    - 如果是卸载，再随机选择一个辅助节点。
    - 计算并记录决策对应的完成时间。
    """
    start_time = time.time()

    solution = {
        'x_i': {},
        'Y_ij': {},
        'energies': {},
        'utilities': {}
    }
    task_completion_times = []
    client_utilities = []

    print("\nRunning 'Random Offload' Baseline Algorithm...")
    print("----------------------------------------------------------")

    # 初始化 Y_ij 矩阵为 0
    for cp in client_params_list:
        for hp in helper_params_list:
            solution['Y_ij'][(cp.id, hp.id)] = 0

    # 1. 为每个客户端独立做出随机决策
    for cp in client_params_list:
        # 创建一个包含所有可能选项的列表
        # 选项 'local' 代表本地执行
        # 选项 'helper_X' 代表卸载到 ID 为 X 的辅助节点
        decision_options = ['local'] + [f'helper_{hp.id}' for hp in helper_params_list]
        
        # 随机选择一个决策
        chosen_option = random.choice(decision_options)

        # 2. 根据决策计算结果
        if chosen_option == 'local':
            # --- 本地执行 ---
            solution['x_i'][cp.id] = 1
            
            f_B_local = cp.F_loc_i
            completion_time = calculate_T_B_value(cp.D_i, cp.q_i_B, f_B_local)
            energy = calculate_E_B_value(cp.D_i, cp.q_i_B, f_B_local, cp.k_i)
            utility = -completion_time
            
            task_completion_times.append(completion_time)
            client_utilities.append(utility)
            solution['energies'][cp.id] = energy
            solution['utilities'][cp.id] = utility

            print(f"Client {cp.id}: Chose LOCAL. Time={completion_time:.4f}s, Energy={energy:.4f}, Utility={utility:.4f}")
        
        else: # --- 卸载到某个辅助节点 ---
            solution['x_i'][cp.id] = 0
            
            # 从选项字符串中解析出辅助节点 ID
            helper_id = int(chosen_option.split('_')[1])
            hp = next((h for h in helper_params_list if h.id == helper_id), None)

            if hp:
                solution['Y_ij'][(cp.id, hp.id)] = 1
                
                # 与 greedy 算法类似的假设
                p_i_fixed = 1.0
                N_i_fixed = network_params.N_f
                f_L_local = cp.F_loc_i

                t_encrypt = calculate_T_encrypt_value(cp.D_i, cp.q_i_L, f_L_local)
                t_transmit = calculate_T_transmit_value(cp.D_i, cp.a_i, p_i_fixed)
                t_compute = calculate_T_compute_helper_value(cp.D_i, hp.q_j_r, hp.F_max_j)
                
                completion_time = t_encrypt + t_transmit + t_compute
                
                e_encrypt = calculate_E_encrypt_value(cp.D_i, cp.q_i_L, f_L_local, cp.k_i_encrypt)
                e_transmit = calculate_E_transmit_value(cp.D_i, cp.a_i, p_i_fixed)
                energy = e_encrypt + e_transmit
                utility = calculate_U_ij_value(
                    a_i=cp.a_i,
                    p_i_val=p_i_fixed,
                    N_i_val=N_i_fixed,
                    N_f_param=network_params.N_f,
                    S_j=hp.S_j,
                    n_attacker_density_at_j=hp.n_attacker_density_at_j
                )

                task_completion_times.append(completion_time)
                client_utilities.append(utility)
                solution['energies'][cp.id] = energy
                solution['utilities'][cp.id] = utility

                print(f"Client {cp.id}: Chose OFFLOAD to Helper {hp.id}. Time={completion_time:.4f}s, Energy={energy:.4f}, Utility={utility:.4f}")

    # 3. 汇总系统最终结果
    # 最终目标是所有客户端效用中的最小值
    final_objective = min(client_utilities) if client_utilities else 0
    makespan = max(task_completion_times) if task_completion_times else 0
    
    elapsed_time = time.time() - start_time

    print("----------------------------------------------------------")
    print(f"'Random Offload' baseline finished in {elapsed_time:.4f} seconds.")
    print(f"Final System Objective (negative makespan): {final_objective:.4f}")
    print(f"Final System Performance (Makespan): {makespan:.4f} seconds.")

    return final_objective, solution, makespan