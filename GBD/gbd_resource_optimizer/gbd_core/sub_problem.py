# gbd_resource_optimizer/gbd_core/sub_problem.py
# -*- coding: utf-8 -*-
"""
构建和求解 GBD 子问题 (Primal Problem) 和松弛子问题。
使用命名约束 (setattr 或 add_component) 代替 ConstraintList。
松弛子问题中，所有可能导致不可行的约束都被软化。
变量的边界和初始值已根据参数归一化进行调整。
"""

import sys
import os
import pyomo.environ as pe
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.core import Suffix
import logging

# --- 1. 导入模块 (使用绝对路径) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gbd_resource_optimizer.problem_def import (
    calculate_T_B, calculate_E_B, calculate_T_encrypt, 
    calculate_T_transmit, calculate_T_compute_helper,
    calculate_E_encrypt, calculate_E_transmit,
    calculate_U_ij_expression
)


def build_sub_problem_model(fixed_int_vars, client_params_list, helper_params_list, network_params):
    """
    构建 GBD 子问题 (Primal Problem) 的 Pyomo 模型。
    """
    model = pe.ConcreteModel(name="GBD_SubProblem")

    client_indices = [cp.id for cp in client_params_list]
    helper_indices = [hp.id for hp in helper_params_list]
    client_params_map = {cp.id: cp for cp in client_params_list}
    helper_params_map = {hp.id: hp for hp in helper_params_list}

    # --- Pyomo 变量 (连续变量，使用归一化后的单位) ---
    # 功率 p_i (W)
    model.p = pe.Var(client_indices, within=pe.NonNegativeReals, bounds=(0, 1.0), initialize=0.1)
    # 频率 f (GHz)
    huge_freq = 20.0 # 20 GHz, a reasonable large upper bound
    model.f_B = pe.Var(client_indices, within=pe.NonNegativeReals, bounds=(0, huge_freq), initialize=1.0)
    model.f_L = pe.Var(client_indices, within=pe.NonNegativeReals, bounds=(0, huge_freq), initialize=1.0)
    model.f_j_r_assigned = pe.Var(client_indices, helper_indices, within=pe.NonNegativeReals, bounds=(0, huge_freq), initialize=1.0)
    # 辅助变量 zeta
    model.zeta = pe.Var(within=pe.Reals, initialize=0.0)

    model.objective = pe.Objective(expr=-model.zeta, sense=pe.minimize)

    # --- 约束 ---
    for i in client_indices:
        cp = client_params_map[i]
        x_i_fixed = fixed_int_vars.get('x_i', {}).get(i, 1)
        N_i_fixed = fixed_int_vars.get('N_i', {}).get(i, 1)

        T_B_expr = calculate_T_B(cp.D_i, cp.q_i_B, model.f_B[i])
        E_B_expr = calculate_E_B(cp.D_i, cp.q_i_B, model.f_B[i], cp.k_i)
        T_encrypt_expr = calculate_T_encrypt(cp.D_i, cp.q_i_L, model.f_L[i])
        E_encrypt_expr = calculate_E_encrypt(cp.D_i, cp.q_i_L, model.f_L[i], cp.k_i_encrypt)
        T_transmit_expr = calculate_T_transmit(cp.D_i, cp.a_i, model.p[i])
        E_transmit_expr = calculate_E_transmit(cp.D_i, cp.a_i, model.p[i])

        T_compute_helper_sum_expr = sum(
            int(fixed_int_vars.get('Y_ij', {}).get((i, j), 0)) *
            calculate_T_compute_helper(cp.D_i, helper_params_map[j].q_j_r, model.f_j_r_assigned[i, j])
            for j in helper_indices
        ) if helper_indices else 0.0
        
        sum_Yij_Uij_expr = sum(
            int(fixed_int_vars.get('Y_ij', {}).get((i, j), 0)) *
            calculate_U_ij_expression(
                a_i=cp.a_i, p_i_var=model.p[i], N_i_var=N_i_fixed, N_f_param=network_params.N_f,
                S_j=helper_params_map[j].S_j,
                n_attacker_density_at_j=helper_params_map[j].n_attacker_density_at_j
            )
            for j in helper_indices
        ) if helper_indices else 0.0

        T_R_expr = T_encrypt_expr + T_transmit_expr + T_compute_helper_sum_expr
        E_R_expr = E_encrypt_expr + E_transmit_expr

        model.add_component(f'C2_latency_{i}', pe.Constraint(expr=x_i_fixed * T_B_expr + (1 - x_i_fixed) * T_R_expr <= cp.T_max_i))
        model.add_component(f'C3_energy_{i}', pe.Constraint(expr=x_i_fixed * E_B_expr + (1 - x_i_fixed) * E_R_expr <= cp.E_max_i))
        model.add_component(f'C4_fB_{i}', pe.Constraint(expr=model.f_B[i] <= cp.F_loc_i))
        model.add_component(f'C4_fL_{i}', pe.Constraint(expr=model.f_L[i] <= cp.F_loc_i))

        O_i_expr = x_i_fixed * (-T_B_expr) + (1 - x_i_fixed) * sum_Yij_Uij_expr
        model.add_component(f'constraint_c10_{i}', pe.Constraint(expr=model.zeta <= O_i_expr))

    for j in helper_indices:
        hp = helper_params_map[j]
        total_freq_demand_on_helper = sum(
            int(fixed_int_vars.get('Y_ij', {}).get((i_other, j), 0)) * model.f_j_r_assigned[i_other, j]
            for i_other in client_indices
        )
        model.add_component(f'C5_helper_capacity_{j}', pe.Constraint(expr=total_freq_demand_on_helper <= hp.F_max_j))

    model.dual = Suffix(direction=Suffix.IMPORT)
    return model


def solve_sub_problem(pyomo_model, solver_name='ipopt', tee=False):
    """求解 GBD 子问题 Pyomo 模型。"""
    solver = SolverFactory(solver_name)
    results = solver.solve(pyomo_model, tee=tee)

    status = results.solver.status
    term_cond = results.solver.termination_condition

    objective_value, continuous_solution, dual_variables = None, None, None

    if term_cond == TerminationCondition.optimal or term_cond == TerminationCondition.locallyOptimal:
        objective_value = pe.value(pyomo_model.objective)
        continuous_solution = {v_name: {idx: pe.value(var_obj[idx]) for idx in var_obj} if var_obj.is_indexed() else pe.value(var_obj) 
                               for v_name, var_obj in pyomo_model.component_map(pe.Var).items()}

        dual_variables = {}
        c10_duals = {}
        for i in pyomo_model.p.index_set():
            constraint_obj = getattr(pyomo_model, f'constraint_c10_{i}')
            try:
                c10_duals[i] = pyomo_model.dual.get(constraint_obj, 0.0)
            except KeyError:
                 logging.warning(f"Could not retrieve dual for constraint_c10_{i}.")
                 c10_duals[i] = 0.0
        dual_variables['c10'] = c10_duals
    
    elif term_cond == TerminationCondition.infeasible:
        logging.warning("Subproblem is infeasible.")
    else:
        logging.error(f"Subproblem solver failed. Status: {status}, Condition: {term_cond}")

    return status, term_cond, objective_value, continuous_solution, dual_variables


def build_relaxed_sub_problem_model(fixed_int_vars, client_params_list, helper_params_list, network_params):
    """构建 GBD 松弛子问题 (Feasibility Problem) 的 Pyomo 模型。"""
    model = pe.ConcreteModel(name="GBD_RelaxedSubProblem")

    client_indices = [cp.id for cp in client_params_list]
    helper_indices = [hp.id for hp in helper_params_list]
    client_params_map = {cp.id: cp for cp in client_params_list}
    helper_params_map = {hp.id: hp for hp in helper_params_list}

    # --- Pyomo 变量 (使用归一化后的单位和边界) ---
    model.p = pe.Var(client_indices, within=pe.NonNegativeReals, bounds=(0, 1.0), initialize=0.1)
    huge_freq = 20.0 # 20 GHz
    model.f_B = pe.Var(client_indices, within=pe.NonNegativeReals, bounds=(0, huge_freq), initialize=1.0)
    model.f_L = pe.Var(client_indices, within=pe.NonNegativeReals, bounds=(0, huge_freq), initialize=1.0)
    model.f_j_r_assigned = pe.Var(client_indices, helper_indices, within=pe.NonNegativeReals, bounds=(0, huge_freq), initialize=1.0)
    model.zeta = pe.Var(within=pe.Reals, initialize=0.0)

    # 松弛变量
    model.H_relax = pe.Var(client_indices, within=pe.NonNegativeReals, initialize=0.0)
    model.Q_relax = pe.Var(client_indices, within=pe.NonNegativeReals, initialize=0.0)
    model.Y_hat_relax = pe.Var(client_indices, within=pe.NonNegativeReals, initialize=0.0)
    model.F_B_relax = pe.Var(client_indices, within=pe.NonNegativeReals, initialize=0.0)
    model.F_L_relax = pe.Var(client_indices, within=pe.NonNegativeReals, initialize=0.0)
    model.F_H_relax = pe.Var(helper_indices, within=pe.NonNegativeReals, initialize=0.0)

    # --- 目标函数：最小化所有松弛变量之和 ---
    model.objective = pe.Objective(
        expr=sum(model.H_relax[i] + model.Q_relax[i] + model.Y_hat_relax[i] + model.F_B_relax[i] + model.F_L_relax[i] for i in client_indices) +
             sum(model.F_H_relax[j] for j in helper_indices),
        sense=pe.minimize
    )

    # --- 约束 ---
    for i in client_indices:
        cp = client_params_map[i]
        x_i_fixed = fixed_int_vars.get('x_i', {}).get(i, 1)
        N_i_fixed = fixed_int_vars.get('N_i', {}).get(i, 1)

        # 表达式计算 (与上面相同)
        T_B_expr = calculate_T_B(cp.D_i, cp.q_i_B, model.f_B[i])
        E_B_expr = calculate_E_B(cp.D_i, cp.q_i_B, model.f_B[i], cp.k_i)
        T_encrypt_expr = calculate_T_encrypt(cp.D_i, cp.q_i_L, model.f_L[i])
        E_encrypt_expr = calculate_E_encrypt(cp.D_i, cp.q_i_L, model.f_L[i], cp.k_i_encrypt)
        T_transmit_expr = calculate_T_transmit(cp.D_i, cp.a_i, model.p[i])
        E_transmit_expr = calculate_E_transmit(cp.D_i, cp.a_i, model.p[i])

        
        T_compute_helper_sum_expr = sum(
            int(fixed_int_vars.get('Y_ij', {}).get((i, j), 0)) *
            calculate_T_compute_helper(cp.D_i, helper_params_map[j].q_j_r, model.f_j_r_assigned[i, j])
            for j in helper_indices
        ) if helper_indices else 0.0
        
        
        sum_Yij_Uij_expr = sum(
            int(fixed_int_vars.get('Y_ij', {}).get((i, j), 0)) *
            calculate_U_ij_expression(
                a_i=cp.a_i, p_i_var=model.p[i], N_i_var=N_i_fixed, N_f_param=network_params.N_f,
                S_j=helper_params_map[j].S_j,
                n_attacker_density_at_j=helper_params_map[j].n_attacker_density_at_j
            )
            for j in helper_indices
        ) if helper_indices else 0.0

        T_R_expr = T_encrypt_expr + T_transmit_expr + T_compute_helper_sum_expr
        E_R_expr = E_encrypt_expr + E_transmit_expr

        # --- 添加松弛约束 ---
        model.add_component(f'C2_relaxed_{i}', pe.Constraint(expr=x_i_fixed * T_B_expr + (1 - x_i_fixed) * T_R_expr <= cp.T_max_i + model.H_relax[i]))
        model.add_component(f'C3_relaxed_{i}', pe.Constraint(expr=x_i_fixed * E_B_expr + (1 - x_i_fixed) * E_R_expr <= cp.E_max_i + model.Q_relax[i]))
        
        model.add_component(f'C4_fB_relaxed_{i}', pe.Constraint(expr=model.f_B[i] <= cp.F_loc_i + model.F_B_relax[i]))
        model.add_component(f'C4_fL_relaxed_{i}', pe.Constraint(expr=model.f_L[i] <= cp.F_loc_i + model.F_L_relax[i]))

        O_i_expr = x_i_fixed * (-T_B_expr) + (1 - x_i_fixed) * sum_Yij_Uij_expr
        model.add_component(f'constraint_c10_relaxed_{i}', pe.Constraint(expr=model.zeta <= O_i_expr + model.Y_hat_relax[i]))

    # --- 松弛 C5 ---
    for j in helper_indices:
        hp = helper_params_map[j]
    
        total_freq_demand_on_helper = sum(
            int(fixed_int_vars.get('Y_ij', {}).get((i_other, j), 0)) * model.f_j_r_assigned[i_other, j]
            for i_other in client_indices
        )
        model.add_component(f'C5_relaxed_{j}', pe.Constraint(expr=total_freq_demand_on_helper <= hp.F_max_j + model.F_H_relax[j]))

    model.dual = Suffix(direction=Suffix.IMPORT)
    return model


def solve_relaxed_sub_problem(pyomo_model, solver_name='ipopt', tee=False):
    """求解 GBD 松弛子问题 Pyomo 模型。"""
    solver = SolverFactory(solver_name)
    results = solver.solve(pyomo_model, tee=tee)

    status = results.solver.status
    term_cond = results.solver.termination_condition

    objective_value, continuous_solution, dual_variables = None, None, None

    if term_cond == TerminationCondition.optimal or term_cond == TerminationCondition.locallyOptimal:
        objective_value = pe.value(pyomo_model.objective)
        
        continuous_solution = {v_name: {idx: pe.value(var_obj[idx]) for idx in var_obj} if var_obj.is_indexed() else pe.value(var_obj) 
                               for v_name, var_obj in pyomo_model.component_map(pe.Var).items()}

        dual_variables = {}
        # 提取所有松弛约束的对偶
        for const_name in ['C2_relaxed', 'C3_relaxed', 'constraint_c10_relaxed', 'C4_fB_relaxed', 'C4_fL_relaxed']:
            duals = {}
            for i in pyomo_model.p.index_set():
                full_const_name = f'{const_name}_{i}'
                if hasattr(pyomo_model, full_const_name):
                    try:
                        duals[i] = pyomo_model.dual.get(getattr(pyomo_model, full_const_name), 0.0)
                    except KeyError:
                        duals[i] = 0.0
            dual_variables[const_name] = duals
        
        c5_duals = {}
        for j in pyomo_model.f_j_r_assigned.index_set().get(1, []):
             full_const_name = f'C5_relaxed_{j}'
             if hasattr(pyomo_model, full_const_name):
                try:
                    c5_duals[j] = pyomo_model.dual.get(getattr(pyomo_model, full_const_name), 0.0)
                except KeyError:
                    c5_duals[j] = 0.0
        dual_variables['C5_relaxed'] = c5_duals
    else:
        logging.error(f"Relaxed subproblem solver failed. Status: {status}, Condition: {term_cond}")

    return status, term_cond, objective_value, continuous_solution, dual_variables