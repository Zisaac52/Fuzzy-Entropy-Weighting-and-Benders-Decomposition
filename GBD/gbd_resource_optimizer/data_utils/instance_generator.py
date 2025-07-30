# gbd_resource_optimizer/data_utils/instance_generator.py
# -*- coding: utf-8 -*-
"""
生成用于资源优化问题的实例数据。
包含参数归一化逻辑，以提高优化求解器的数值稳定性。
"""

import random
# import numpy as np # 可选，用于更复杂的分布

# 尝试从父级目录导入
try:
    from gbd_resource_optimizer.problem_def import ClientParams, HelperParams, NetworkParams
except ImportError:
    # Fallback for direct execution
    from ..problem_def import ClientParams, HelperParams, NetworkParams


# 定义参数的基础范围和异构性调整因子 (保持原始单位)
PARAM_CONFIG = {
    'D_i': {'base_min': 100 * 1024 * 8, 'base_max': 200 * 1024 * 8, 'unit': 'bits'}, # 100KB-200KB
    'q_i_B': {'base_min': 500, 'base_max': 1500, 'unit': 'cycles/bit'},
    'q_i_L': {'base_min': 100, 'base_max': 300, 'unit': 'cycles/bit'},
    'k_i': {'base_min': 1e-28, 'base_max': 5e-28, 'unit': 'J*s^3/cycle^3 (effective)'}, # k_i * f^2 is energy per cycle
    'F_loc_i': {'base_min': 1e9, 'base_max': 3e9, 'unit': 'Hz'}, # 1-3 GHz
    'T_max_i': {'base_min': 0.5, 'base_max': 2.0, 'unit': 's'},
    'E_max_i': {'base_min': 0.1, 'base_max': 1.0, 'unit': 'J'},
    'a_i': {'base_min': 1e-5, 'base_max': 1e-4, 'unit': '(bits/s)/sqrt(W)'}, # 这个单位和范围可能需要重新评估
    'k_i_encrypt': {'base_min': 1e-28, 'base_max': 5e-28, 'unit': 'J*s^3/cycle^3 (effective)'},
    'q_j_r': {'base_min': 300, 'base_max': 1000, 'unit': 'cycles/bit'},
    'F_max_j': {'base_min': 10e9, 'base_max': 30e9, 'unit': 'Hz'}, # 10-30 GHz
    'S_j': {'base_min': 0.5, 'base_max': 1.0, 'unit': 'score'},
    'n_attacker_density_at_j': {'base_min': 0.01, 'base_max': 0.2, 'unit': 'density'},
}

HETEROGENEITY_FACTORS = {
    'low': 0.5,     # 较小异构性，参数更集中
    'medium': 1.0,  # 中等异构性，使用基础范围
    'high': 1.5     # 较大异构性，参数更分散
}

CLIENT_TYPE_CONFIG = {
    'strong': {
        'F_loc_i_range': (2.5e9, 4.0e9),  # 强算力 (2.5-4.0 GHz)
        'T_max_i_range': (1.5, 3.0),     # 延迟容忍度高 (1.5-3.0s)
        'E_max_i_range': (0.8, 1.5),     # 能量预算充足 (0.8-1.5J)
    },
    'weak': {
        'F_loc_i_range': (0.8e9, 1.5e9),   # 弱算力 (0.8-1.5 GHz)
        'T_max_i_range': (1.0, 2.0),     # 延迟要求苛刻 (0.3-0.8s) -> 放宽到 (1.0, 2.0)
        'E_max_i_range': (0.5, 1.0),     # 能量预算紧张 (0.5-1.0J) -> 放宽
    }
}

def _get_random_value(param_name, heterogeneity_level):
    """根据参数名称和异构性级别生成随机值。"""
    config = PARAM_CONFIG[param_name]
    base_min = config['base_min']
    base_max = config['base_max']
    
    center = (base_min + base_max) / 2
    base_spread = (base_max - base_min) / 2
    
    factor = HETEROGENEITY_FACTORS.get(heterogeneity_level, 1.0)
    current_spread = base_spread * factor
    
    val_min = center - current_spread
    val_max = center + current_spread

    final_min = max(val_min, 0 if base_min >= 0 else val_min)
    final_max = max(final_min, val_max)

    return random.uniform(final_min, final_max)


def generate_instance(num_clients, num_helpers, heterogeneity_level='medium', strong_client_ratio=0.5, task_workload=None):
    """
    生成一个问题实例，并对参数进行归一化。

    Args:
        num_clients (int): 客户端数量。
        num_helpers (int): 辅助节点数量。
        heterogeneity_level (str): 异构性级别 ('low', 'medium', 'high')。
        task_workload (int, optional): 固定任务数据量 (单位: KB). Defaults to None.

    Returns:
        tuple: (client_params_list, helper_params_list, network_params)
               所有返回的参数都经过了归一化处理。
    """
    if heterogeneity_level not in HETEROGENEITY_FACTORS:
        raise ValueError(f"Invalid heterogeneity_level: {heterogeneity_level}. "
                         f"Must be one of {list(HETEROGENEITY_FACTORS.keys())}")

    client_params_list = []
    num_strong_clients = int(num_clients * strong_client_ratio)
    for i in range(num_clients):
        client_type = 'strong' if i < num_strong_clients else 'weak'
        type_config = CLIENT_TYPE_CONFIG[client_type]
        
        # --- 原有参数生成逻辑 (保留部分) ---
        D_i_orig = _get_random_value('D_i', heterogeneity_level)
        if task_workload is not None:
            # 如果提供了 task_workload (单位: KB)，则固定 D_i 的值 (转换为 bits)
            # 注意：这里的 task_workload 是从命令行传入的，我们约定其单位是 KB
            D_i_orig = task_workload * 1024 * 8
            
        q_i_B_orig = _get_random_value('q_i_B', heterogeneity_level)
        q_i_L_orig = _get_random_value('q_i_L', heterogeneity_level)
        k_i_orig = _get_random_value('k_i', heterogeneity_level)
        k_i_encrypt_orig = _get_random_value('k_i_encrypt', heterogeneity_level)
        
        # --- 使用新的类型化参数范围 ---
        F_loc_i_orig = random.uniform(*type_config['F_loc_i_range'])
        T_max_i_raw = random.uniform(*type_config['T_max_i_range'])
        E_max_i_raw = random.uniform(*type_config['E_max_i_range'])

        # --- 归一化逻辑 (大部分不变) ---
        D_i_norm = D_i_orig / 1e6
        F_loc_i_norm = F_loc_i_orig / 1e9
        q_i_B_norm = q_i_B_orig * 1e-3
        q_i_L_norm = q_i_L_orig * 1e-3
        k_i_norm = k_i_orig * 1e27
        k_i_encrypt_norm = k_i_encrypt_orig * 1e27
        a_i_norm = random.uniform(1.0, 10.0)
        
        client_params_list.append(ClientParams(
            client_id=i,
            D_i=D_i_norm,
            q_i_B=q_i_B_norm,
            q_i_L=q_i_L_norm,
            k_i=k_i_norm,
            F_loc_i=F_loc_i_norm,
            k_i_encrypt=k_i_encrypt_norm,
            a_i=a_i_norm,
            T_max_i=T_max_i_raw, # 直接使用新生成的值
            E_max_i=E_max_i_raw  # 直接使用新生成的值
        ))

    helper_params_list = []
    for j in range(num_helpers):
        # 1. 生成原始尺度的参数
        q_j_r_orig = _get_random_value('q_j_r', heterogeneity_level)
        F_max_j_orig = _get_random_value('F_max_j', heterogeneity_level)
        
        # 2. 进行归一化
        q_j_r_norm = q_j_r_orig * 1e-3  # cycles/bit -> (GHz*s)/Mb
        F_max_j_norm = F_max_j_orig / 1e9 # Hz -> GHz
        
        helper_params_list.append(HelperParams(
            helper_id=j,
            q_j_r=q_j_r_norm,
            F_max_j=F_max_j_norm,
            S_j=_get_random_value('S_j', heterogeneity_level),
            n_attacker_density_at_j=_get_random_value('n_attacker_density_at_j', heterogeneity_level)
        ))
        
    # 区块长度 N_f 的单位是 bits，保持不变
    N_f = random.choice([64, 128]) 
    network_params = NetworkParams(N_f=N_f)
    
    return client_params_list, helper_params_list, network_params

if __name__ == '__main__':
    # 示例用法
    print("--- Generating Normalized Instance (Medium Heterogeneity) ---")
    clients, helpers, network = generate_instance(num_clients=3, num_helpers=2, heterogeneity_level='medium')
    
    for client in clients:
        print(f"Client {client.id}: D_i={client.D_i:.2f} Mb, q_i_B={client.q_i_B:.2f}, k_i={client.k_i:.2e}, F_loc_i={client.F_loc_i:.2f} GHz, a_i={client.a_i:.2f}")
        
    for helper in helpers:
        print(f"Helper {helper.id}: q_j_r={helper.q_j_r:.2f}, F_max_j={helper.F_max_j:.2f} GHz, S_j={helper.S_j:.2f}")
        
    print(f"Network: N_f={network.N_f} bits")