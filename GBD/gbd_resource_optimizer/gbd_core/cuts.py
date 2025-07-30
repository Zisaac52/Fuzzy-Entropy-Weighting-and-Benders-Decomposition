# gbd_resource_optimizer/gbd_core/cuts.py

import pyomo.environ as pe
from pyomo.core.expr.base import ExpressionBase
from typing import Dict, List, Any
import logging

# --- 1. 使用健壮的绝对路径导入 ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gbd_resource_optimizer.problem_def import (
    ClientParams, HelperParams, NetworkParams,
    calculate_O_i_expression_for_cut,
    calculate_O_i_value,
    calculate_T_B_value, calculate_E_B,
    calculate_T_encrypt, calculate_T_transmit, calculate_T_compute_helper,
    calculate_E_encrypt, calculate_E_transmit,
    calculate_U_ij_value,calculate_T_encrypt_value, calculate_T_transmit_value,
    calculate_T_compute_helper_value,calculate_E_encrypt_value, calculate_E_transmit_value
)


def generate_optimality_cut_expression(
    fixed_int_vars: Dict[str, Any],
    sp_obj_val: float,
    sp_dual_vars: Dict[str, Dict[Any, float]],
    sp_continuous_sol: Dict[str, Dict[Any, float]],
    mp_vars: pe.ConcreteModel,
    client_params_list: List[ClientParams],
    helper_params_list: List[HelperParams],
    network_params: NetworkParams
) -> pe.Expression:
    """
    根据子问题的解和对偶变量生成最优性割 (Optimality Cut) 的 Pyomo 表达式。
    形式为: eta >= sp_obj_val + sum(dual_k * (g_k(MP_vars) - g_k(fixed_vars)))
    """
    if sp_obj_val is None:
        raise ValueError("Cannot generate optimality cut: subproblem objective value is None.")

    cut_expr_rhs = sp_obj_val

    # 我们只关心 C10 约束 (zeta <= O_i) 的对偶变量
    if 'c10' in sp_dual_vars and sp_dual_vars['c10'] is not None:
        for cp in client_params_list:
            client_id = cp.id
            dual_c10_i = sp_dual_vars['c10'].get(client_id, 0.0)
            
            if abs(dual_c10_i) > 1e-9:
                # 1. 计算 O_i(MP_vars) 的表达式 (关于主问题变量)
                O_i_expr_MP = calculate_O_i_expression_for_cut(
                    cp, helper_params_list, network_params,
                    mp_vars, sp_continuous_sol, fixed_int_vars
                )
                
                # 2. 计算 O_i(fixed_vars) 的数值 (基于子问题解)
                T_B_val_fixed = calculate_T_B_value(cp.D_i, cp.q_i_B, sp_continuous_sol['f_B'][client_id])
                sum_U_val_fixed = 0.0
                if fixed_int_vars['x_i'][client_id] == 0:
                    for hp in helper_params_list:
                        if fixed_int_vars.get('Y_ij', {}).get((client_id, hp.id), 0) == 1:
                            n_i_fixed_val = fixed_int_vars['N_i'][client_id]
                            p_i_val_for_u = sp_continuous_sol['p'][client_id]
                            sum_U_val_fixed += calculate_U_ij_value(
                                cp.a_i, p_i_val_for_u, n_i_fixed_val,
                                network_params.N_f, hp.S_j, hp.n_attacker_density_at_j
                            )
                O_i_val_at_fixed = calculate_O_i_value(
                    fixed_int_vars['x_i'][client_id], T_B_val_fixed, sum_U_val_fixed
                )
                
                if O_i_val_at_fixed is None:
                    raise ValueError(f"O_i_val_at_fixed is None for client {client_id}.")

                # 3. 添加到割表达式的右侧
                cut_expr_rhs += dual_c10_i * (O_i_expr_MP - O_i_val_at_fixed)
    
    optimality_cut = (mp_vars.eta >= cut_expr_rhs)
    return optimality_cut


def generate_feasibility_cut_expression(
    fixed_int_vars: Dict[str, Any],
    relaxed_sp_obj_val: float, # 用于判断，但不用在割的表达式中
    relaxed_sp_dual_vars: Dict[str, Dict[Any, float]],
    relaxed_sp_continuous_sol: Dict[str, Dict[Any, float]],
    mp_vars: pe.ConcreteModel,
    client_params_list: List[ClientParams],
    helper_params_list: List[HelperParams],
    network_params: NetworkParams,
    tolerance: float = 1e-6
) -> pe.Expression:
    """
    根据松弛子问题的解和对偶变量生成一个强的可行性割 (Feasibility Cut)。
    通过只使用那些在松弛问题中被违反的约束（即对应松弛变量 > tolerance）
    来模拟 IIS (Irreducible Infeasible Subset)，从而生成更强的割。
    """
    # 根据Benders理论，可行性割的RHS应该从松弛子问题的目标值开始
    # relaxed_sp_obj_val 是所有松弛变量的总和，必须 > 0
    # 理论上，可行性割的RHS应为0。松弛目标值>0仅用于触发此逻辑。
    cut_expr_rhs = 0.0

    violated_constraints_found = False

    # 提取所有松弛变量的解
    h_relax_sol = relaxed_sp_continuous_sol.get('H_relax', {})
    q_relax_sol = relaxed_sp_continuous_sol.get('Q_relax', {})
    y_hat_relax_sol = relaxed_sp_continuous_sol.get('Y_hat_relax', {})
    f_b_relax_sol = relaxed_sp_continuous_sol.get('F_B_relax', {}) # 暂不处理，因为不涉及主问题变量
    f_l_relax_sol = relaxed_sp_continuous_sol.get('F_L_relax', {}) # 暂不处理
    f_h_relax_sol = relaxed_sp_continuous_sol.get('F_H_relax', {})
    
    # --- 1. 处理时延约束 C2' (H_relax) ---
    duals_for_c2 = relaxed_sp_dual_vars.get('C2_relaxed', {})
    if duals_for_c2:
        for cp in client_params_list:
            # **核心判断**：只有当这个约束被违反时，才将其加入割
            if h_relax_sol.get(cp.id, 0.0) > tolerance:
                dual_val = duals_for_c2.get(cp.id, 0.0)
                if abs(dual_val) > tolerance:
                    violated_constraints_found = True
                    logging.debug(f"Client {cp.id}: Latency constraint violated (H_relax > {tolerance}). Adding to feasibility cut.")
                    g_k_MP, g_k_fixed = _get_g_k_latency(cp, helper_params_list, network_params, mp_vars, relaxed_sp_continuous_sol, fixed_int_vars)
                    cut_expr_rhs += dual_val * (g_k_MP - g_k_fixed)

    # --- 2. 处理能耗约束 C3' (Q_relax) ---
    duals_for_c3 = relaxed_sp_dual_vars.get('C3_relaxed', {})
    if duals_for_c3:
        for cp in client_params_list:
            # **核心判断**
            if q_relax_sol.get(cp.id, 0.0) > tolerance:
                dual_val = duals_for_c3.get(cp.id, 0.0)
                if abs(dual_val) > tolerance:
                    violated_constraints_found = True
                    logging.debug(f"Client {cp.id}: Energy constraint violated (Q_relax > {tolerance}). Adding to feasibility cut.")
                    g_k_MP, g_k_fixed = _get_g_k_energy(cp, relaxed_sp_continuous_sol, fixed_int_vars, mp_vars)
                    cut_expr_rhs += dual_val * (g_k_MP - g_k_fixed)

    # --- 3. 处理目标函数相关约束 C10' (Y_hat_relax) ---
    duals_for_c10 = relaxed_sp_dual_vars.get('constraint_c10_relaxed', {})
    if duals_for_c10:
        for cp in client_params_list:
            # **核心判断**
            if y_hat_relax_sol.get(cp.id, 0.0) > tolerance:
                dual_val = duals_for_c10.get(cp.id, 0.0)
                if abs(dual_val) > tolerance:
                    violated_constraints_found = True
                    logging.debug(f"Client {cp.id}: Zeta objective constraint violated (Y_hat_relax > {tolerance}). Adding to feasibility cut.")
                    O_i_expr_MP = calculate_O_i_expression_for_cut(cp, helper_params_list, network_params, mp_vars, relaxed_sp_continuous_sol, fixed_int_vars)
                    O_i_val_fixed = calculate_O_i_value_from_sp_sol(cp, helper_params_list, network_params, relaxed_sp_continuous_sol, fixed_int_vars)
                    # g_k(y) = zeta - O_i(y) - Y_hat <= 0. dual*(g_k(MP) - g_k(fixed)) = dual*(-(O_i_MP - O_i_fixed))
                    cut_expr_rhs += dual_val * (-(O_i_expr_MP - O_i_val_fixed))

    # --- 4. 处理辅助节点算力约束 C5' (F_H_relax) ---
    duals_for_c5 = relaxed_sp_dual_vars.get('C5_relaxed', {})
    if duals_for_c5:
        for hp in helper_params_list:
            j = hp.id
            if f_h_relax_sol.get(j, 0.0) > tolerance:
                dual_val = duals_for_c5.get(j, 0.0)
                if abs(dual_val) > tolerance:
                    violated_constraints_found = True
                    logging.debug(f"Helper {j}: Capacity constraint violated (F_H_relax > {tolerance}). Adding to feasibility cut.")
                    g_k_MP, g_k_fixed = _get_g_k_capacity(hp, client_params_list, mp_vars, relaxed_sp_continuous_sol, fixed_int_vars)
                    cut_expr_rhs += dual_val * (g_k_MP - g_k_fixed)


    # --- 构建最终的割表达式 ---
    if not violated_constraints_found:
        logging.warning("Feasibility cut is empty. No violated constraints found with duals > tolerance. Returning a slack cut.")
        # 返回一个恒成立的约束，避免给主问题增加无效的割
        return pe.Constraint.Feasible

    feasibility_cut = (0 >= cut_expr_rhs)
    return feasibility_cut


# --- 新增的辅助函数，以减少代码重复 ---
def _get_g_k_latency(cp, helpers, network, mp_vars, sp_sol, fixed_vars):
    T_B_val = calculate_T_B_value(cp.D_i, cp.q_i_B, sp_sol['f_B'][cp.id])
    T_R_val = calculate_T_R_value(cp, helpers, network, sp_sol, fixed_vars)
    g_k_MP = mp_vars.x_i[cp.id] * T_B_val + (1 - mp_vars.x_i[cp.id]) * T_R_val - cp.T_max_i
    g_k_fixed = fixed_vars['x_i'][cp.id] * T_B_val + (1 - fixed_vars['x_i'][cp.id]) * T_R_val - cp.T_max_i
    return g_k_MP, g_k_fixed

def _get_g_k_energy(cp, sp_sol, fixed_vars, mp_vars):
    E_B_val = calculate_E_B_value(cp.D_i, cp.q_i_B, sp_sol['f_B'][cp.id], cp.k_i)
    E_R_val = calculate_E_R_value(cp, sp_sol)
    g_k_MP = mp_vars.x_i[cp.id] * E_B_val + (1 - mp_vars.x_i[cp.id]) * E_R_val - cp.E_max_i
    g_k_fixed = fixed_vars['x_i'][cp.id] * E_B_val + (1 - fixed_vars['x_i'][cp.id]) * E_R_val - cp.E_max_i
    return g_k_MP, g_k_fixed

def _get_g_k_capacity(hp, client_params_list, mp_vars, sp_sol, fixed_vars):
    """构建与C5辅助节点算力约束相关的 g_k(y) 和 g_k(y_hat)"""
    
    # g_k(y) (关于主问题变量 Y_ij)
    # f_j_r 的值来自子问题解，对于割来说是常数
    demand_expr_mp = sum(
        mp_vars.Y_ij[cp.id, hp.id] *
        sp_sol['f_j_r_assigned'][(cp.id, hp.id)]
        for cp in client_params_list
    )
    g_k_MP = demand_expr_mp - hp.F_max_j

    # g_k(y_hat) (数值)
    demand_val_fixed = sum(
        fixed_vars.get('Y_ij', {}).get((cp.id, hp.id), 0) *
        sp_sol['f_j_r_assigned'][(cp.id, hp.id)]
        for cp in client_params_list
    )
    g_k_fixed = demand_val_fixed - hp.F_max_j
    
    return g_k_MP, g_k_fixed

# --- 辅助函数，以减少代码重复 ---
def calculate_T_R_value(client_params, helper_params_list, network_params, sp_sol, fixed_int_vars):
    """根据子问题解计算卸载总时间的数值"""
    cp = client_params
    T_encrypt_val = calculate_T_encrypt_value(cp.D_i, cp.q_i_L, sp_sol['f_L'][cp.id])
    T_transmit_val = calculate_T_transmit_value(cp.D_i, cp.a_i, sp_sol['p'][cp.id])
    
    T_compute_helper_sum_val = 0.0
    if helper_params_list:
        for hp in helper_params_list:
            if fixed_int_vars.get('Y_ij', {}).get((cp.id, hp.id), 0) == 1:
                 T_compute_helper_sum_val += calculate_T_compute_helper_value(
                     cp.D_i, hp.q_j_r, sp_sol['f_j_r_assigned'][(cp.id, hp.id)]
                 )
    return T_encrypt_val + T_transmit_val + T_compute_helper_sum_val

def calculate_E_B_value(D_i, q_i_B, f_i_B_val, k_i):
    """计算本地计算能耗的数值"""
    return k_i * D_i * q_i_B * (f_i_B_val ** 2)

def calculate_E_R_value(client_params, sp_sol):
    """根据子问题解计算卸载能耗的数值"""
    cp = client_params
    E_encrypt_val = calculate_E_encrypt_value(cp.D_i, cp.q_i_L, sp_sol['f_L'][cp.id], cp.k_i_encrypt)
    E_transmit_val = calculate_E_transmit_value(cp.D_i, cp.a_i, sp_sol['p'][cp.id])
    return E_encrypt_val + E_transmit_val

def calculate_O_i_value_from_sp_sol(cp, helper_params_list, network_params, sp_sol, fixed_int_vars):
    """根据子问题解和固定整数解计算 O_i 的数值"""
    T_B_val = calculate_T_B_value(cp.D_i, cp.q_i_B, sp_sol['f_B'][cp.id])
    sum_U_val = 0.0
    if fixed_int_vars['x_i'][cp.id] == 0:
        for hp in helper_params_list:
            if fixed_int_vars.get('Y_ij', {}).get((cp.id, hp.id), 0) == 1:
                sum_U_val += calculate_U_ij_value(
                    cp.a_i, sp_sol['p'][cp.id], fixed_int_vars['N_i'][cp.id],
                    network_params.N_f, hp.S_j, hp.n_attacker_density_at_j
                )
    return calculate_O_i_value(fixed_int_vars['x_i'][cp.id], T_B_val, sum_U_val)