# gbd_resource_optimizer/problem_def.py
# -*- coding: utf-8 -*-
"""
定义与资源优化问题相关的参数类和核心计算函数。
所有以 "_expression" 结尾的函数返回 Pyomo 表达式，用于模型构建。
所有以 "_value" 结尾的函数返回 Python 数值，用于日志记录和后处理。
"""

import pyomo.environ as pe
import math

# 用于避免除零等数值问题的常量
EPSILON = 1e-9
VERY_LARGE_NUMBER = 1e20

# =========================================================================
# 1. 参数定义类
# =========================================================================

class ClientParams:
    """存储客户端相关参数的类。"""
    def __init__(self, client_id, D_i, q_i_B, q_i_L, k_i, F_loc_i, T_max_i, E_max_i, a_i, k_i_encrypt):
        self.id = client_id
        self.D_i = D_i  # 任务数据量 (bits)
        self.q_i_B = q_i_B # 本地计算每 bit CPU 周期数
        self.q_i_L = q_i_L # 加密每 bit CPU 周期数
        self.k_i = k_i  # 本地计算能量系数
        self.F_loc_i = F_loc_i # 本地最大 CPU 频率 (Hz)
        self.T_max_i = T_max_i # 最大延迟容忍 (s)
        self.E_max_i = E_max_i # 最大能量预算 (J)
        self.a_i = a_i # 信道系数 (对应论文 alpha_i)
        self.k_i_encrypt = k_i_encrypt # 加密能量系数

class HelperParams:
    """存储辅助节点相关参数的类。"""
    def __init__(self, helper_id, q_j_r, F_max_j, S_j, n_attacker_density_at_j):
        self.id = helper_id
        self.q_j_r = q_j_r # 辅助节点计算每 bit CPU 周期数 (对应论文 q_i^r)
        self.F_max_j = F_max_j # 辅助节点最大总计算能力 (频率, Hz)
        self.S_j = S_j # 辅助节点能力评分 (用于 U_ij)
        self.n_attacker_density_at_j = n_attacker_density_at_j # 攻击者密度 (对应论文 n_e^j)

class NetworkParams:
    """存储全局网络或任务参数的类。"""
    def __init__(self, N_f):
        self.N_f = N_f # 最大区块长度 (bits)


# =========================================================================
# 2. 返回 Pyomo 表达式的计算函数 (用于模型构建)
# =========================================================================

def calculate_T_B(D_i, q_i_B, f_i_B):
    """计算本地计算时间 T_B 的 Pyomo 表达式。"""
    return pe.Expr_if(IF=f_i_B <= EPSILON, THEN=VERY_LARGE_NUMBER, ELSE=D_i * q_i_B / f_i_B)

def calculate_E_B(D_i, q_i_B, f_i_B, k_i):
    """计算本地计算能耗 E_B 的 Pyomo 表达式。"""
    return k_i * D_i * q_i_B * (f_i_B ** 2)

def calculate_T_encrypt(D_i, q_i_L, f_i_L):
    """计算加密时间 T_encrypt 的 Pyomo 表达式。"""
    return pe.Expr_if(IF=f_i_L <= EPSILON, THEN=VERY_LARGE_NUMBER, ELSE=D_i * q_i_L / f_i_L)

def calculate_T_transmit(D_i, a_i, p_i):
    """计算传输时间 T_transmit 的 Pyomo 表达式。"""
    if not isinstance(a_i, (int, float)) or a_i <= EPSILON:
        return pe.Expression(expr=VERY_LARGE_NUMBER)
    return pe.Expr_if(IF=p_i <= EPSILON, THEN=VERY_LARGE_NUMBER, ELSE=D_i / (a_i * pe.sqrt(p_i)))

def calculate_T_compute_helper(D_i, q_j_r, f_j_r):
    """计算辅助节点计算时间 T_compute 的 Pyomo 表达式。"""
    return pe.Expr_if(IF=f_j_r <= EPSILON, THEN=VERY_LARGE_NUMBER, ELSE=D_i * q_j_r / f_j_r)

def calculate_E_encrypt(D_i, q_i_L, f_i_L, k_i_encrypt):
    """计算加密能耗 E_encrypt 的 Pyomo 表达式。"""
    return k_i_encrypt * D_i * q_i_L * (f_i_L ** 2)

def calculate_E_transmit(D_i, a_i, p_i):
    """计算传输能耗 E_transmit 的 Pyomo 表达式。"""
    if not isinstance(a_i, (int, float)) or a_i <= EPSILON:
        return pe.Expression(expr=VERY_LARGE_NUMBER)
    return pe.Expr_if(IF=p_i <= EPSILON, THEN=0.0, ELSE=D_i * pe.sqrt(p_i) / a_i)

def calculate_U_ij_expression(a_i, p_i_var, N_i_var, N_f_param, S_j, n_attacker_density_at_j):
    """
    计算效用函数 U_ij 的 Pyomo 表达式。
    此表达式对于 N_i 变量是非线性的，不应直接用于线性主问题。
    它主要用于构建非线性的子问题。
    """
    # 检查 N_i_var 是否为0，以避免 log(0)
    # 因为 N_i 是整数变量，所以检查它是否大于0
    is_N_i_zero = (N_i_var <= EPSILON)

    # 避免 log(0)
    log_term_arg = 8 * N_i_var
    log_term = pe.log(log_term_arg) / pe.log(2)

    # 通信增益部分
    # 避免除以零
    comm_gain_denom = 8 * N_i_var
    comm_gain_part = pe.Expr_if(
        IF=comm_gain_denom <= EPSILON,
        THEN=0.0,
        ELSE=(a_i * pe.sqrt(p_i_var) / comm_gain_denom) * log_term
    )

    # 安全增益部分
    # 避免除以零
    base = pe.Expr_if(
        IF=N_f_param <= EPSILON,
        THEN=0.0,
        ELSE=N_i_var / N_f_param
    )
    security_gain = base ** n_attacker_density_at_j

    # 组合成最终表达式
    # 如果 N_i 是0，则整个效用为0
    utility_expr = pe.Expr_if(
        IF=is_N_i_zero,
        THEN=0.0,
        ELSE=comm_gain_part * security_gain * S_j
    )

    return utility_expr

# def calculate_O_i_expression_for_cut(client_params, helper_params_list, network_params,
#                                      mp_vars, sp_sol_cont, fixed_int_vars):
#     """
#     为生成Benders割，计算 O_i 的 Pyomo 表达式。
#     这个表达式是关于【主问题变量】的。
#     其中依赖于子问题连续变量的部分，使用子问题的【解出的数值】。
#     """
#     cp = client_params
    
#     # 1. 计算 -T_B 项：它在割中是常数，其值由子问题的解 f_B 决定
#     f_B_val = sp_sol_cont['f_B'][cp.id]
#     T_B_val = calculate_T_B_value(cp.D_i, cp.q_i_B, f_B_val)
    
#     # # 2. 计算 sum(Y_ij * U_ij) 项：它是关于主问题变量 Y_ij, N_i 和子问题变量 p_i 的表达式
#     # sum_Yij_Uij_expr = pe.Expression(expr=0.0)
#     # if helper_params_list:
#     #     for hp in helper_params_list:
#     #         # U_ij 的计算依赖于 p_i, N_i。在割中，p_i 的值是固定的 (来自子问题解)，
#     #         # 而 N_i 是主问题变量。由于我们简化了 U_ij 使其不依赖 N_i 变量，
#     #         # 这里的 U_ij 只依赖于 p_i，因此在割中也变成一个常数。
#     #         p_i_val = sp_sol_cont['p'][cp.id]
#     #         U_ij_val = calculate_U_ij_value(
#     #             cp.a_i, p_i_val, mp_vars.N_i[cp.id].value, # 即使简化了，这里也用.value获取当前值
#     #             network_params.N_f, hp.S_j, hp.n_attacker_density_at_j
#     #         )
#     #         # 乘以主问题的变量 Y_ij
#     #         sum_Yij_Uij_expr += mp_vars.Y_ij[cp.id, hp.id] * U_ij_val

#      # 2. 计算 sum(Y_ij * U_ij) 项
#     sum_Yij_Uij_expr = pe.Expression(expr=0.0)
#     if helper_params_list:
#         for hp in helper_params_list:
#             p_i_val = sp_sol_cont['p'][cp.id]
            
#             # ========================= 核心修改点 =========================
#             # U_ij 对于割来说是常数，因为我们简化了它对 N_i 的依赖。
#             # 它的值应该用 fixed_int_vars 中的 N_i 来计算，而不是 mp_vars.N_i.value
#             n_i_fixed_val = fixed_int_vars['N_i'][cp.id]
            
#             U_ij_val = calculate_U_ij_value(
#                 cp.a_i, p_i_val, n_i_fixed_val, # <-- 使用来自 fixed_int_vars 的 N_i
#                 network_params.N_f, hp.S_j, hp.n_attacker_density_at_j
#             )
#             # ==========================================================

#             sum_Yij_Uij_expr += mp_vars.Y_ij[cp.id, hp.id] * U_ij_val

#     # 3. 组合成 O_i 表达式
#     # ========================= 核心修改点 =========================
#     # 原来的非线性表达式:
#     # O_i_expr = mp_vars.x_i[cp.id] * (-T_B_val) + (1 - mp_vars.x_i[cp.id]) * sum_Yij_Uij_expr

#     # 修正后的线性表达式:
#     # 利用约束 sum_j Y_ij = 1 - x_i, 我们知道 (1 - x_i) * sum(...) 等价于 sum(...)
#     # 但为了构建割，我们需要一个包含 x_i 的表达式。
#     # O_i = x_i * (-T_B) + (1-x_i) * sum(...)
#     # 让我们重新审视一下这个表达式。
#     # 当 x_i=1, O_i = -T_B_val.
#     # 当 x_i=0, O_i = sum(Y_ij * U_ij_val).
#     # 这本身就是一个分段线性的函数，可以直接用 x_i 和 Y_ij 表达，它是线性的。
#     # 问题在于 pyomo 在处理 (1-x_i) * sum(...) 时，将其识别为了非线性。

#     # 让我们来确认一下 sum_Yij_Uij_expr 是否真的被当作非线性。
#     # U_ij_val 是一个常数。所以 sum_Yij_Uij_expr 是线性的。
#     # (1 - x_i) 是线性的。
#     # 线性 * 线性 = 非线性。这就是问题所在。

#     # 我们使用等价关系 sum_j Y_ij = 1 - x_i 来替换 (1 - x_i)
#     # O_i = x_i * (-T_B) + (sum_j Y_ij_MP[j]) * sum_j(Y_ij_MP[j] * U_ij_val)  <-- 这更复杂了
    
#     # 回到最简单的等价替换：
#     # O_i = x_i * (-T_B_val) + sum_j(Y_ij_MP[j] * U_ij_val)
#     # 我们来验证这个新的线性表达式：
#     # case 1: x_i = 1 (本地). Y_ij 都为0.
#     #    原公式 O_i = 1 * (-T_B_val) + 0 * (...) = -T_B_val
#     #    新公式 O_i = 1 * (-T_B_val) + 0 = -T_B_val  (正确)
#     # case 2: x_i = 0 (卸载). sum_j Y_ij = 1.
#     #    原公式 O_i = 0 * (...) + 1 * sum(Y_ij*U_ij) = sum(Y_ij*U_ij)
#     #    新公式 O_i = 0 * (-T_B_val) + sum(Y_ij*U_ij) = sum(Y_ij*U_ij) (正确)

#     # 结论：这个替换是完全等价且线性的！

#     O_i_expr_linear = mp_vars.x_i[cp.id] * (-T_B_val) + sum_Yij_Uij_expr

#     # ==========================================================
    
#     # return O_i_expr_linear
#     return pe.Expression(expr=O_i_expr_linear)

def calculate_O_i_expression_for_cut(client_params, helper_params_list, network_params,
                                     mp_vars, sp_sol_cont, fixed_int_vars):
    """
    为生成Benders割，计算 O_i 的 Pyomo 表达式。
    返回一个【线性】的 Pyomo 表达式。
    核心思想：在生成割时，U_ij 的值被当作一个常数。这个常数值是在子问题解 (sp_sol_cont for p_i)
    和主问题当前整数解 (fixed_int_vars for N_i) 的点上计算出来的。
    """
    cp = client_params
    
    # 1. 计算 -T_B 项 (常数, 基于子问题解)
    f_B_val = sp_sol_cont.get('f_B', {}).get(cp.id, 0.0)
    T_B_val = calculate_T_B_value(cp.D_i, cp.q_i_B, f_B_val)
    
    # 2. 计算 sum(Y_ij * U_ij_const) 项
    sum_Yij_Uij_expr = pe.Expression(expr=0.0)
    if helper_params_list:
        terms = []
        # 从子问题解中获取 p_i 的值
        p_i_val = sp_sol_cont.get('p', {}).get(cp.id, 0.0)
        
        # 从上一个主问题的整数解中获取 N_i 的值
        # 这是实现“在当前整数解点 N_i_fixed 处进行线性化”的关键
        n_i_fixed_val = fixed_int_vars.get('N_i', {}).get(cp.id, 1) # 默认值为1以防万一
            
        for hp in helper_params_list:
            # 计算 U_ij 的【数值】，此时 N_i 被视为常数
            U_ij_val = calculate_U_ij_value(
                cp.a_i, p_i_val, n_i_fixed_val,
                network_params.N_f, hp.S_j, hp.n_attacker_density_at_j
            )
            
            # 将此常数值乘以主问题的变量 Y_ij，这是一个线性项
            if abs(U_ij_val) > EPSILON:
                terms.append(mp_vars.Y_ij[cp.id, hp.id] * U_ij_val)

        if terms:
            sum_Yij_Uij_expr = sum(terms)

    # 3. 组合成 O_i 表达式 (使用线性等价形式)
    # O_i = x_i * (-T_B) + sum_j(Y_ij * U_ij_const)
    # 这个表达式对于主问题变量 x_i 和 Y_ij 是线性的。
    O_i_expr_linear = mp_vars.x_i[cp.id] * (-T_B_val) + sum_Yij_Uij_expr
    
    return O_i_expr_linear
# =========================================================================
# 3. 返回 Python 数值的计算函数 (用于日志记录和后处理)
# =========================================================================

def calculate_T_B_value(D_i, q_i_B, f_i_B_val):
    """计算本地计算时间的数值。"""
    if f_i_B_val <= EPSILON:
        return VERY_LARGE_NUMBER
    return D_i * q_i_B / f_i_B_val

def calculate_U_ij_value(a_i, p_i_val, N_i_val, N_f_param, S_j, n_attacker_density_at_j):
    """计算效用函数 U_ij 的数值。"""
    p_i_val = float(p_i_val)
    N_i_val = float(N_i_val)

    if N_i_val <= EPSILON or N_f_param <= EPSILON or a_i <= EPSILON or S_j < 0 or p_i_val < 0:
        return 0.0

    try:
        log_term_arg = 8 * N_i_val
        if log_term_arg <= 0: return 0.0
        log_term = math.log2(log_term_arg)

        comm_gain = (a_i * math.sqrt(p_i_val) / (8 * N_i_val)) * log_term if p_i_val > EPSILON else 0.0

        base = N_i_val / N_f_param
        if base < 0: return 0.0
        security_gain = math.pow(base, n_attacker_density_at_j)
    except (ValueError, OverflowError):
        return 0.0

    utility = comm_gain * security_gain * S_j
    return utility if math.isfinite(utility) else 0.0


def calculate_O_i_value(x_i_val, T_B_val, sum_Yij_Uij_val):
    """计算目标函数 O_i 的数值。"""
    if int(x_i_val) == 1:
        return -T_B_val
    elif int(x_i_val) == 0:
        return sum_Yij_Uij_val
    else:
        raise ValueError(f"Invalid value for x_i: {x_i_val}. Must be 0 or 1.")
    

# =========================================================================
# 4. (新增) 返回数值的辅助计算函数
# =========================================================================

def calculate_T_encrypt_value(D_i, q_i_L, f_i_L_val):
    if f_i_L_val <= EPSILON: return VERY_LARGE_NUMBER
    return D_i * q_i_L / f_i_L_val

def calculate_T_transmit_value(D_i, a_i, p_i_val):
    if p_i_val <= EPSILON or a_i <= EPSILON: return VERY_LARGE_NUMBER
    return D_i / (a_i * math.sqrt(p_i_val))

def calculate_T_compute_helper_value(D_i, q_j_r, f_j_r_val):
    if f_j_r_val <= EPSILON: return VERY_LARGE_NUMBER
    return D_i * q_j_r / f_j_r_val

def calculate_E_B_value(D_i, q_i_B, f_i_B_val, k_i):
    """计算本地计算能耗的数值。"""
    return k_i * D_i * q_i_B * (f_i_B_val ** 2)

def calculate_E_encrypt_value(D_i, q_i_L, f_i_L_val, k_i_encrypt):
    return k_i_encrypt * D_i * q_i_L * (f_i_L_val ** 2)

def calculate_E_transmit_value(D_i, a_i, p_i_val):
    if a_i <= EPSILON: return VERY_LARGE_NUMBER
    if p_i_val <= EPSILON: return 0.0
    return D_i * math.sqrt(p_i_val) / a_i