# gbd_resource_optimizer/baselines/greedy_offload.py

import time
from gbd_resource_optimizer.problem_def import (
    calculate_T_B_value, calculate_E_B_value,
    calculate_T_encrypt_value, calculate_T_transmit_value, calculate_T_compute_helper_value,
    calculate_E_encrypt_value, calculate_E_transmit_value, calculate_U_ij_value
)

def solve_greedy_offload(client_params_list, helper_params_list, network_params):
    """
    根据参考论文重构的“贪心卸载”基准算法。

    核心逻辑：遍历每一个客户端，独立地为它做出最大化自身效用 O_i 的决策。
    - 本地效用 O_i = -T_B_i
    - 卸载效用 O_i = U_ij
    决策基于比较 本地效用 vs. 卸载到各个辅助节点的效用。
    """
    start_time = time.time()

    solution = {
        'x_i': {},
        'Y_ij': {},
        'energies': {},
        'utilities': {}
    }
    client_utilities = []
    task_completion_times = []

    print("\nRunning 'Greedy Offload' Baseline Algorithm (Refactored)...")
    print("----------------------------------------------------------")

    # --- 1. 为每个客户端独立做出贪心决策 ---
    for cp in client_params_list:
        # --- a. 计算本地执行的效用 ---
        # 假设本地计算使用其最大计算能力
        f_B_local = cp.F_loc_i
        local_time = calculate_T_B_value(cp.D_i, cp.q_i_B, f_B_local)
        local_utility = -local_time
        local_energy = calculate_E_B_value(cp.D_i, cp.q_i_B, f_B_local, cp.k_i)
        
        # (decision_type, utility, completion_time, energy, helper_id)
        best_option = ('local', local_utility, local_time, local_energy, None)

        # --- b. 遍历所有辅助节点，计算卸载到每个节点的效用 ---
        for hp in helper_params_list:
            # 假设1: 客户端使用固定的发射功率
            p_i_fixed = 1.0
            
            # 假设2: 客户端使用网络定义的最大区块长度
            N_i_fixed = network_params.N_f

            # 假设3: 卸载时，加密过程在本地以最大能力完成
            f_L_local = cp.F_loc_i
            
            # 计算卸载效用 U_ij
            # 计算卸载任务的完成时间 (Makespan)
            # T_offload = T_encrypt + T_transmit + T_compute_helper
            t_encrypt = calculate_T_encrypt_value(cp.D_i, cp.q_i_L, f_L_local)
            t_transmit = calculate_T_transmit_value(cp.D_i, cp.a_i, p_i_fixed)
            # 假设该客户端独占辅助节点的全部计算资源
            t_compute = calculate_T_compute_helper_value(cp.D_i, hp.q_j_r, hp.F_max_j)
            offload_time = t_encrypt + t_transmit + t_compute
            offload_utility = calculate_U_ij_value(
                a_i=cp.a_i,
                p_i_val=p_i_fixed,
                N_i_val=N_i_fixed,
                N_f_param=network_params.N_f,
                S_j=hp.S_j,
                n_attacker_density_at_j=hp.n_attacker_density_at_j
            )
            
            # 计算卸载能耗
            e_encrypt = calculate_E_encrypt_value(cp.D_i, cp.q_i_L, f_L_local, cp.k_i_encrypt)
            e_transmit = calculate_E_transmit_value(cp.D_i, cp.a_i, p_i_fixed)
            offload_energy = e_encrypt + e_transmit

            # --- c. 比较并选择效用最大的选项 ---
            if offload_utility > best_option[1]:
                best_option = ('offload', offload_utility, offload_time, offload_energy, hp.id)

        # --- d. 记录此客户端的最终决策 ---
        final_decision, final_utility, final_time, final_energy, best_helper_id = best_option
        
        client_utilities.append(final_utility)
        task_completion_times.append(final_time)
        solution['energies'][cp.id] = final_energy
        solution['utilities'][cp.id] = final_utility

        if final_decision == 'local':
            solution['x_i'][cp.id] = 1
            for hp in helper_params_list:
                solution['Y_ij'][(cp.id, hp.id)] = 0
            print(f"Client {cp.id}: Chose LOCAL. Utility={final_utility:.4f}, Time={final_time:.4f}s, Energy={final_energy:.4f}")
        else:  # offload
            solution['x_i'][cp.id] = 0
            for hp in helper_params_list:
                solution['Y_ij'][(cp.id, hp.id)] = 1 if hp.id == best_helper_id else 0
            print(f"Client {cp.id}: Chose OFFLOAD to Helper {best_helper_id}. Utility={final_utility:.4f}, Time={final_time:.4f}s, Energy={final_energy:.4f}")

    # --- 2. 汇总系统最终结果 ---
    # 最终目标是所有客户端效用中的最小值
    final_objective = min(client_utilities) if client_utilities else 0
    
    # 最终时间是所有任务完成时间中的最大值 (Makespan)
    makespan = max(task_completion_times) if task_completion_times else 0
    
    elapsed_time = time.time() - start_time

    print("----------------------------------------------------------")
    print(f"'Greedy Offload' baseline finished in {elapsed_time:.4f} seconds.")
    print(f"Final System Objective (min utility): {final_objective:.4f}")
    print(f"Final System Performance (Makespan): {makespan:.4f} seconds.")

    # 返回与GBD优化器兼容的格式
    # 注意：这里的 total_time 返回的是 Makespan，而不是算法执行时间
    return final_objective, solution, makespan
