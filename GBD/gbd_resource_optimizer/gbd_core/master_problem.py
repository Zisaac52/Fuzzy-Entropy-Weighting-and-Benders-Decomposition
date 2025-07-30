import pyomo.environ as pe
import pyomo.opt as po
import math
from gbd_resource_optimizer.problem_def import calculate_T_compute_helper, VERY_LARGE_NUMBER


def build_master_problem_model(client_params_list, helper_params_list, network_params):
    """
    Builds the Pyomo model for the master problem.

    Args:
        client_params_list (list): List of client parameters.
        helper_params_list (list): List of helper parameters.
        network_params (object): Network parameters object, expected to have N_f attribute.

    Returns:
        pe.ConcreteModel: The constructed Pyomo model for the master problem.
    """
    model = pe.ConcreteModel(name="MasterProblem")

    # Indices
    num_clients = len(client_params_list)
    num_helpers = len(helper_params_list)
    model.client_indices = pe.Set(initialize=range(num_clients))
    model.helper_indices = pe.Set(initialize=range(num_helpers))

    # Variables
    # x_i: 1 if client i is selected, 0 otherwise
    model.x_i = pe.Var(model.client_indices, within=pe.Binary, doc="Client selection variable")

    # N_i: Number of federated learning rounds for client i
    # bounds based on network_params.N_f as per (C7) and problem description
    model.N_i = pe.Var(model.client_indices, within=pe.PositiveIntegers,
                       bounds=(1, network_params.N_f), doc="Number of FL rounds for client i")

    # Y_ij: 1 if client i is assigned to helper j, 0 otherwise
    model.Y_ij = pe.Var(model.client_indices, model.helper_indices, within=pe.Binary,
                        doc="Client to helper assignment variable")

    # eta: Objective variable for the master problem (corresponds to y_0 or zeta in the paper)
    model.eta = pe.Var(within=pe.Reals, bounds=(-1e20, None), doc="Master problem objective variable (with a large negative lower bound to prevent unboundedness in initial iterations)")

    # Objective Function (17)
    # minimize eta
     # ========================= 核心修改点 =========================
    # 修改目标函数，加入一个关于 N_i 的微小激励项
    
    # 一个非常小的权重，以确保主目标仍然是优化 eta
    w_N = 1e-6 
    
    # 新的目标函数表达式
    objective_expr = model.eta - w_N * sum(model.N_i[i] for i in model.client_indices)
    
    model.objective = pe.Objective(expr=objective_expr, sense=pe.minimize)
    # ==========================================================
    # model.objective = pe.Objective(expr=model.eta, sense=pe.minimize)

    # Initial Constraints
    # (C1) 0 <= x_i <= 1 is implicitly handled by pe.Binary for model.x_i

    # (C7) 0 <= N_i <= N_f is handled by pe.NonNegativeIntegers and bounds for model.N_i

    # (C8) sum_j Y_ij = 1 - x_i for each client i
    # This constraint ensures that if a client is not selected (x_i = 0), it must be assigned to exactly one helper.
    # If a client is selected (x_i = 1), it is not assigned to any helper (sum_j Y_ij = 0).
    # This interpretation aligns with the idea that Y_ij represents assignment for *offloaded* computation.
    # If the problem implies selected clients can also be "assisted" by helpers for some tasks,
    # the constraint might need adjustment based on the precise meaning of Y_ij.
    # Assuming Y_ij is for clients *not* performing local FL (i.e., x_i=0).
    # def c8_rule(m, i):
    #     return sum(m.Y_ij[i, j] for j in m.helper_indices) == 1 - m.x_i[i]
    # model.c8_sum_Yij = pe.Constraint(model.client_indices, rule=c8_rule, doc="Constraint C8: sum_j Y_ij = 1 - x_i")

    # (C9) 0 <= Y_ij <= 1 is implicitly handled by pe.Binary for model.Y_ij
    # ========================= 核心修改点 =========================
    # 确保 C8 约束被正确添加和激活

    # (C8) sum_j Y_ij = 1 - x_i for each client i
    def c8_rule(m, i):
        # 如果没有辅助节点，这个约束没有意义，并且会导致 sum 一个空列表
        if not m.helper_indices:
            # 如果没有辅助节点，那么必须本地计算，即 x_i=1，Y_ij必须为0
            return m.x_i[i] == 1
        else:
            return sum(m.Y_ij[i, j] for j in m.helper_indices) == 1 - m.x_i[i]
    
    # 将约束添加到模型中
    model.c8_sum_Yij = pe.Constraint(model.client_indices, rule=c8_rule, doc="Constraint C8: Link x_i and Y_ij")
    # ==========================================================

 # ========================= 核心修改点 =========================
    # 添加约束 C5 的线性化版本，以防止 MP 生成愚蠢的解

    # 为每个辅助节点 j 添加一个容量约束
    def c5_linearized_rule(m, j):
        helper_p = helper_params_list[j] # 假设列表索引与id一致
        
        # 计算每个客户端 i 如果卸载到 j，所需的最小频率
        # f_ij_r >= D_i * q_j_r / T_max_i  (这里我们忽略了 T_encrypt 和 T_transmit)
        # 这是一个简化但有效的估计
        total_min_freq_demand = 0.0
        for i in m.client_indices:
            client_p = client_params_list[i]
            
            # 忽略 T_encrypt 和 T_transmit 后的最大允许计算时间
            # 修复：将 T_tx_encrypt_estimate 从一个硬编码的较大值（0.1）
            # 替换为一个更合理的下界估计。
            # 这里的逻辑是，我们必须为数据传输和加密过程保留一个最小的、
            # 非零的时间余量。我们不能假设计算时间可以占据整个 T_max_i。
            # 选择一个很小的值（如 1e-3）可以在避免过度约束导致不必要
            # 的不可行性的同时，仍然有效地剪掉那些分配了不切实际的
            # 计算时间的解（即 T_compute 几乎等于 T_max_i 的解）。
            # 步骤 2a: 估算最小加密时间 T_encrypt_min
            # T_encrypt_min = D_i * q_i_L / f_i_L, 使用客户端最大加密频率 F_loc_i
            T_encrypt_min = client_p.D_i * client_p.q_i_L / client_p.F_loc_i

            # 步骤 2b: 估算最小传输时间 T_transmit_min
            # T_transmit_min = D_i / (a_i * sqrt(p_i)), 使用一个估计的最大传输功率
            p_max_estimate = 0.5  # 瓦特, 一个合理的固定上限
            T_transmit_min = client_p.D_i / (client_p.a_i * math.sqrt(p_max_estimate))

            # 步骤 2c: 更新总估算时间
            T_tx_encrypt_estimate = T_encrypt_min + T_transmit_min
            max_allowed_T_compute = client_p.T_max_i - T_tx_encrypt_estimate
            
            # 计算最小所需频率
            if max_allowed_T_compute > 1e-9:
                f_ij_min = (client_p.D_i * helper_p.q_j_r) / max_allowed_T_compute
            else:
                f_ij_min = VERY_LARGE_NUMBER # 如果时间预算几乎为0，则需要无穷大频率
            
            # 只有当客户端 i 卸载到 j 时，这个需求才会被计算在内
            # 我们需要线性化 (1 - x_i) * Y_ij
            # 引入辅助变量 Z_ij = (1-x_i) * Y_ij
            # 这里为了简单，我们先用一个不完全等价但线性的近似：
            # (1 - x_i) 和 Y_ij 都必须为1，才会计入。这等价于 Z_ij。
            # 我们用 Y_ij 来近似 Z_ij，因为如果 Y_ij=1，x_i 必须为0。
            total_min_freq_demand += m.Y_ij[i, j] * f_ij_min

        return total_min_freq_demand <= helper_p.F_max_j

    # 将这个新约束添加到模型中
    model.c5_linearized_capacity = pe.Constraint(model.helper_indices, rule=c5_linearized_rule, doc="Linearized Helper Capacity Constraint")
    # ==========================================================

    model.benders_cuts_list = pe.ConstraintList(doc="List to store Benders cuts")
    return model




def add_benders_cut_to_master(master_model, cut_pyomo_expr):
    """
    Adds a Benders cut to the master problem model.

    Args:
        master_model (pe.ConcreteModel): The master problem model instance.
        cut_pyomo_expr (pe.Expression): The Pyomo expression representing the Benders cut.
    """
    master_model.benders_cuts_list.add(cut_pyomo_expr)

def solve_master_problem(pyomo_model, solver_name='cbc', tee=False):
    """
    Solves the given Pyomo model (master problem).

    Args:
        pyomo_model (pe.ConcreteModel): The Pyomo model to be solved.
        solver_name (str): Name of the solver to use (e.g., 'cbc', 'glpk').
        tee (bool): If True, display solver output.

    Returns:
        tuple: (objective_value, solution_variables, solver_status, termination_condition)
               objective_value (float or None): Value of the objective function if solved.
               solution_variables (dict or None): Dictionary of variable values if solved.
               solver_status (pyomo.opt.SolverStatus): Solver status enum.
               termination_condition (pyomo.opt.TerminationCondition): Termination condition enum.
    """
    solver = po.SolverFactory(solver_name)
    results = solver.solve(pyomo_model, tee=tee)

    status = results.solver.status
    term_cond = results.solver.termination_condition

    obj_val = None
    # Initialize solution_variables with empty dictionaries for each variable type
    solution_variables = {'x_i': {}, 'N_i': {}, 'Y_ij': {}}

    if term_cond == po.TerminationCondition.optimal or \
       term_cond == po.TerminationCondition.feasible:
        obj_val = pe.value(pyomo_model.eta)
        for i in pyomo_model.client_indices:
            # For x_i:
            # Only add to solution_variables if x_i[i].value is not None.
            # This ensures downstream .get(key, default_value) works correctly.
            x_i_val = pyomo_model.x_i[i].value
            if x_i_val is not None:
                solution_variables['x_i'][i] = int(round(x_i_val))

            # For N_i:
            # Only add to solution_variables if N_i[i].value is not None.
            # If it's None, the key 'i' will be missing from solution_variables['N_i'],
            # which is the desired behavior to allow downstream .get(key, default_value) to work correctly.
            n_i_val = pyomo_model.N_i[i].value
            if n_i_val is not None:
                solution_variables['N_i'][i] = int(round(n_i_val))
            
            for j in pyomo_model.helper_indices:
                # For Y_ij:
                # Only add to solution_variables if Y_ij[i,j].value is not None.
                # This ensures downstream .get(key, default_value) works correctly.
                y_ij_val = pyomo_model.Y_ij[i,j].value
                if y_ij_val is not None:
                    solution_variables['Y_ij'][(i,j)] = int(round(y_ij_val))


    elif term_cond == po.TerminationCondition.infeasible:
        obj_val = float('inf')  # For minimization problems
        # solution_variables remains as initialized (empty dicts or with None)
        print(f"Master problem is infeasible. Status: {status}, Condition: {term_cond}")

    elif term_cond == po.TerminationCondition.unbounded:
        obj_val = float('-inf') # For minimization problems
        # solution_variables remains as initialized (empty dicts or with None)
        # Potentially, some variable values might be available but are part of an unbounded ray.
        # For simplicity, we are not extracting them here.
        print(f"Master problem is unbounded. Status: {status}, Condition: {term_cond}")
    
    else:
        # Handle other termination conditions (e.g., maxTimeLimit, error, etc.)
        # obj_val remains None
        # solution_variables remains as initialized
        print(f"Solver in solve_master_problem from master_problem.py terminated with unhandled status: {status}, condition: {term_cond}")

    return obj_val, solution_variables, status, term_cond
