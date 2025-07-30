# gbd_resource_optimizer/baselines/local_first.py

import time
from ..problem_def import calculate_T_B_value, calculate_O_i_value, calculate_E_B_value

def solve_local_first(client_params_list, helper_params_list, network_params):
    """
    实现“本地优先”基准算法。
    
    在此算法中，所有客户端都强制进行本地计算。
    目标是计算与 GBD 相同的目标函数：max(min(O_i))，
    在本地优先场景下，即为 min(-T_B_i)。
    """
    start_time = time.time()
    
    client_utilities = []
    local_computation_times = []
    energies = {}
    utilities = {}
    
    # --- 决策变量 ---
    solution = {
        'x_i': {},
        'Y_ij': {},
        'N_i': {},
        'O_i': {},
        'energies': {},
        'utilities': {}
    }
    
    print("\nRunning 'Local First' Baseline Algorithm...")
    print("---------------------------------------------")
    
    # --- 1. 计算每个客户端的效用 ---
    for cp in client_params_list:
        client_id = cp.id
        
        # 决策：强制本地计算
        solution['x_i'][client_id] = 1
        solution['N_i'][client_id] = 0 # 本地计算不涉及区块构建，N_i为0
        
        # 计算本地计算时间 T_B
        # 注意：本地计算使用其自身的最大频率 F_loc_i
        local_computation_time = calculate_T_B_value(cp.D_i, cp.q_i_B, cp.F_loc_i)
        local_computation_times.append(local_computation_time)

        # 计算本地能耗 E_B
        energy = calculate_E_B_value(cp.D_i, cp.q_i_B, cp.F_loc_i, cp.k_i)
        
        # 决策：不卸载到任何帮助者
        sum_Yij_Uij = 0.0
        if helper_params_list:
            for hp in helper_params_list:
                solution['Y_ij'][(client_id, hp.id)] = 0
        
        # 计算该客户端的效用 O_i
        # 根据 problem_def, 当 x_i=1, O_i = -T_B
        utility = calculate_O_i_value(1, local_computation_time, sum_Yij_Uij)
        
        solution['O_i'][client_id] = utility
        solution['energies'][client_id] = energy
        solution['utilities'][client_id] = utility
        client_utilities.append(utility)
        
        print(f"Client {client_id}: Forced local. Time={local_computation_time:.4f}, Energy={energy:.4f}, Utility={utility:.4f}")
            
    # --- 2. 汇总结果 ---
    # total_time 是所有客户端本地计算时间中的最大值
    total_time = max(local_computation_times) if local_computation_times else 0.0
    
    # 目标值是所有客户端效用中的最小值，等价于 -max(T_B_i)
    final_objective = min(client_utilities) if client_utilities else 0.0
    
    print("---------------------------------------------")
    print(f"'Local First' baseline finished. Performance Time (max(T_B_i)): {total_time:.4f} seconds.")
    print(f"Final Objective (min(O_i)): {final_objective:.4f}")
    
    # 返回一个与GBD类似的输出格式，方便统一处理
    return final_objective, solution, total_time
