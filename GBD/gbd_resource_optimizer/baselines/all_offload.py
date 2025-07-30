# gbd_resource_optimizer/baselines/all_offload.py

import time
from gbd_resource_optimizer.problem_def import (
    calculate_T_encrypt_value, calculate_T_transmit_value, calculate_T_compute_helper_value,
    calculate_E_encrypt_value, calculate_E_transmit_value, calculate_U_ij_value
)

def solve_all_offload(client_params_list, helper_params_list, network_params):
    """
    "全部卸载" 基准算法。
    该算法强制所有客户端将任务卸载到第一个可用的辅助节点 (helpers[0])。
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
    
    # 检查是否有可用的辅助节点
    if not helper_params_list:
        raise ValueError("No helpers available for offloading.")

    # --- 1. 指定目标辅助节点 ---
    target_helper = helper_params_list[0]
    
    print("\nRunning 'All Offload' Baseline Algorithm...")
    print("----------------------------------------------------------")
    print(f"All tasks will be offloaded to Helper {target_helper.id}.")

    # --- 2. 遍历所有客户端并计算卸载时间 ---
    for cp in client_params_list:
        # 假设1: 客户端使用固定的发射功率
        p_i_fixed = 1.0
        
        # 假设2: 客户端使用网络定义的最大区块长度
        N_i_fixed = network_params.N_f

        # 假设3: 加密过程在本地以最大能力完成
        f_L_local = cp.F_loc_i
        
        # 计算卸载到 target_helper 的总完成时间
        # T_offload = T_encrypt + T_transmit + T_compute_helper
        t_encrypt = calculate_T_encrypt_value(cp.D_i, cp.q_i_L, f_L_local)
        t_transmit = calculate_T_transmit_value(cp.D_i, cp.a_i, p_i_fixed)
        # 假设该客户端独占目标辅助节点的全部计算资源
        t_compute = calculate_T_compute_helper_value(cp.D_i, target_helper.q_j_r, target_helper.F_max_j)
        
        offload_time = t_encrypt + t_transmit + t_compute
        task_completion_times.append(offload_time)

        # 计算卸载能耗 E_r = E_encrypt + E_transmit
        e_encrypt = calculate_E_encrypt_value(cp.D_i, cp.q_i_L, f_L_local, cp.k_i_encrypt)
        e_transmit = calculate_E_transmit_value(cp.D_i, cp.a_i, p_i_fixed)
        total_energy = e_encrypt + e_transmit
        
        # 对于卸载，效用为 -T_r
        utility = calculate_U_ij_value(
            a_i=cp.a_i,
            p_i_val=p_i_fixed,
            N_i_val=N_i_fixed,
            N_f_param=network_params.N_f,
            S_j=target_helper.S_j,
            n_attacker_density_at_j=target_helper.n_attacker_density_at_j
        )
        client_utilities.append(utility)

        # --- 3. 记录此客户端的决策和指标 ---
        solution['x_i'][cp.id] = 0
        solution['energies'][cp.id] = total_energy
        solution['utilities'][cp.id] = utility
        
        for hp in helper_params_list:
            solution['Y_ij'][(cp.id, hp.id)] = 1 if hp.id == target_helper.id else 0
        
        print(f"Client {cp.id}: Offloaded to Helper {target_helper.id}. Time={offload_time:.4f}s, Energy={total_energy:.4f}, Utility={utility:.4f}")

    # --- 4. 汇总系统最终结果 ---
    # 最终目标是所有任务完成时间负值的最小值（等价于最大化最差效用）
    # 由于所有任务都卸载，效用为负的完成时间
    final_objective = min(client_utilities) if client_utilities else 0
    
    # 最终时间是所有任务完成时间中的最大值 (Makespan)
    makespan = max(task_completion_times) if task_completion_times else 0
    
    elapsed_time = time.time() - start_time

    print("----------------------------------------------------------")
    print(f"'All Offload' baseline finished in {elapsed_time:.4f} seconds.")
    print(f"Final System Objective (min utility): {final_objective:.4f}")
    print(f"Final System Performance (Makespan): {makespan:.4f} seconds.")

    # 返回与GBD优化器兼容的格式
    return final_objective, solution, makespan