import subprocess
import random
import re
import os
import time

# --- 配置参数 ---
MAIN_SCRIPT_PATH = "gbd_resource_optimizer/main.py" # 用于文件操作（如读写种子）
MAIN_MODULE_NAME = "gbd_resource_optimizer.main"   # 用于 python -m 执行
# START_SEED = 0 # 不再需要顺序起始种子
MAX_SEED_ATTEMPTS = 1000 # 您之前设置为200，这里根据文件内容是1000，如果需要改回请告知
PYTHON_EXECUTABLE = "python" # 或者您的 python3.11 解释器的完整路径
MIN_RANDOM_SEED = 0
MAX_RANDOM_SEED = 2**32 - 1 # 常见的随机种子上限

# --- 辅助函数 ---

def get_main_script_path():
    """获取 main.py 的绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设此脚本位于 gbd 项目的根目录下，MAIN_SCRIPT_PATH 是相对路径
    return os.path.join(current_dir, MAIN_SCRIPT_PATH)

def read_params_from_main_script(script_path):
    """从 main.py 中读取 num_clients 和 num_helpers"""
    num_clients = None
    num_helpers = None
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            clients_match = re.search(r"num_clients\s*=\s*(\d+)", content)
            if clients_match:
                num_clients = int(clients_match.group(1))
            
            helpers_match = re.search(r"num_helpers\s*=\s*(\d+)", content)
            if helpers_match:
                num_helpers = int(helpers_match.group(1))
                
    except FileNotFoundError:
        print(f"错误: 脚本 {script_path} 未找到。")
    except Exception as e:
        print(f"读取参数时出错: {e}")
    return num_clients, num_helpers

def update_seed_in_main_script(script_path, new_seed):
    """修改 main.py 中的 RANDOM_SEED 值"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        found_seed = False
        with open(script_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if re.match(r"^\s*RANDOM_SEED\s*=\s*\d+", line):
                    f.write(f"RANDOM_SEED = {new_seed}\n")
                    found_seed = True
                elif re.match(r"#\s*RANDOM_SEED\s*=\s*\d+", line): # 如果是被注释掉的
                    f.write(f"RANDOM_SEED = {new_seed}\n") # 取消注释并更新
                    found_seed = True
                else:
                    f.write(line)
        if not found_seed:
            print(f"警告: 未在 {script_path} 中找到 RANDOM_SEED 行进行更新。")
            return False
        return True
    except FileNotFoundError:
        print(f"错误: 脚本 {script_path} 未找到。")
        return False
    except Exception as e:
        print(f"更新种子时出错: {e}")
        return False

def check_success_from_output(output_str):
    """根据输出来判断执行是否成功"""
    critical_failure_flags = [
        "Algorithm terminated prematurely.",
        "No feasible solution was found during the iterations.",
        "Restoration Failed!",
        "ValueError: Cannot load a SolverResults object with bad status: error",
        "Master problem could not be solved",
        "Subproblem solver failed",
        "Relaxed subproblem could not be solved",
        "Stopping GBD.",
        "Master problem is infeasible",
        "Traceback (most recent call last):", 
        "Error:",
        "Ipopt 3.14.17\\x3a Converged to a locally infeasible point. Problem may be infeasible."
    ]

    if any(flag in output_str for flag in critical_failure_flags):
        return False

    if "Best Feasible Solution Found:" in output_str:
        return True

    if "Algorithm Converged" in output_str:
        ub_match_best = re.search(r"Final Upper Bound \(Best UB Found\):\s*([-\w.]+)", output_str)
        ub_match_final = re.search(r"Final Upper Bound \(UB\):\s*([-\w.]+)", output_str)
        ub_value_str = None
        if ub_match_best: ub_value_str = ub_match_best.group(1)
        elif ub_match_final: ub_value_str = ub_match_final.group(1)
        
        if ub_value_str and ub_value_str.lower() != "infinity" and ub_value_str.lower() != "inf":
            try:
                float(ub_value_str) 
                return True 
            except ValueError:
                return False 
        else:
            return False 
    return False

# --- 主逻辑 ---
if __name__ == "__main__":
    main_script_file_path = get_main_script_path() # Path for file operations
    if not os.path.exists(main_script_file_path):
        print(f"主脚本文件 {main_script_file_path} 不存在，请检查 MAIN_SCRIPT_PATH 配置。")
        exit(1)

    print(f"将随机尝试最多 {MAX_SEED_ATTEMPTS} 个不同的随机种子 (范围: {MIN_RANDOM_SEED}-{MAX_RANDOM_SEED}) 来运行模块 {MAIN_MODULE_NAME}...")
    
    initial_num_clients, initial_num_helpers = read_params_from_main_script(main_script_file_path)
    if initial_num_clients is None or initial_num_helpers is None:
        print("无法从主脚本读取 num_clients 或 num_helpers，退出。")
        exit(1)

    found_successful_seed = False
    successful_seed = -1
    tested_seeds = set() # 用于存储已测试的种子
    
    original_main_script_content = ""
    try:
        with open(main_script_file_path, 'r', encoding='utf-8') as f:
            original_main_script_content = f.read()
    except Exception as e:
        print(f"无法读取原始 main.py 内容: {e}")
        exit(1)

    for attempt in range(MAX_SEED_ATTEMPTS):
        # 生成唯一的随机种子
        current_seed_value = random.randint(MIN_RANDOM_SEED, MAX_RANDOM_SEED)
        retry_count = 0
        while current_seed_value in tested_seeds and retry_count < 100: # 最多重试100次找新种子
            current_seed_value = random.randint(MIN_RANDOM_SEED, MAX_RANDOM_SEED)
            retry_count += 1
        if current_seed_value in tested_seeds:
            print("未能生成唯一的随机种子，可能已尝试了大部分范围或遇到罕见情况，停止。")
            break
        tested_seeds.add(current_seed_value)

        print(f"\n--- 尝试第 {attempt + 1}/{MAX_SEED_ATTEMPTS} 次，随机种子: {current_seed_value} ---")
        
        if not update_seed_in_main_script(main_script_file_path, current_seed_value):
            print("更新种子失败，跳过此次尝试。")
            continue
            
        current_num_clients, current_num_helpers = read_params_from_main_script(main_script_file_path)
        if current_num_clients is None or current_num_helpers is None:
             print("在更新种子后无法读取参数，使用初始值。")
             current_num_clients = initial_num_clients
             current_num_helpers = initial_num_helpers

        print(f"正在使用种子 {current_seed_value} (客户端: {current_num_clients}, 辅助节点: {current_num_helpers}) 运行模块 {MAIN_MODULE_NAME}...")
        
        try:
            # 工作目录应为项目根目录，以便 python -m 能正确找到模块
            project_root_dir = os.path.dirname(os.path.abspath(__file__)) # 假设 find_solvable_seed.py 在根目录
            # 如果 find_solvable_seed.py 不在根目录，需要调整 project_root_dir 的获取方式
            # 例如: project_root_dir = os.path.dirname(os.path.dirname(main_script_file_path)) 
            # (如果 MAIN_SCRIPT_PATH 是 gbd_resource_optimizer/main.py 这样的形式)

            process = subprocess.run(
                [PYTHON_EXECUTABLE, "-m", MAIN_MODULE_NAME], # 修改为模块化运行
                capture_output=True,
                text=False, 
                timeout=300, 
                cwd=project_root_dir # 确保从项目根目录运行
            )
            
            stdout_str = process.stdout.decode('utf-8', errors='ignore')
            stderr_str = process.stderr.decode('utf-8', errors='ignore')

            is_successful_run = check_success_from_output(stdout_str)
            
            has_critical_stderr = False
            if stderr_str:
                critical_stderr_patterns = ["Traceback (most recent call last):", "Error:", "Exception:"]
                if any(pattern in stderr_str for pattern in critical_stderr_patterns):
                     if not is_successful_run and any(flag in stderr_str for flag in critical_failure_flags):
                        has_critical_stderr = True
                     elif "WARNING" not in stderr_str.upper(): 
                        has_critical_stderr = True

            if is_successful_run and not has_critical_stderr:
                print(f"\n成功！找到可求解的种子。")
                print(f"  成功种子: {current_seed_value}")
                print(f"  客户端数量: {current_num_clients}")
                print(f"  辅助节点数量: {current_num_helpers}")
                found_successful_seed = True
                successful_seed = current_seed_value
                break
            else:
                print(f"种子 {current_seed_value} 未能成功求解。")
                if stderr_str and (has_critical_stderr or not is_successful_run): # Print stderr if critical or if stdout didn't indicate success
                    print(f"  错误/输出信息摘要 (stderr):\n{stderr_str[:1000]}")
                if not is_successful_run and stdout_str : 
                     print(f"  输出信息摘要 (stdout) 未包含成功标志或包含失败标志:\n{stdout_str[:1000]}")

        except subprocess.TimeoutExpired:
            print(f"种子 {current_seed_value} 运行超时。")
        except Exception as e:
            print(f"运行主脚本时发生错误 (种子 {current_seed_value}): {e}")
        
    if not found_successful_seed:
        print(f"\n在随机尝试了 {attempt + 1 if attempt < MAX_SEED_ATTEMPTS else MAX_SEED_ATTEMPTS} 个种子后 (范围: {MIN_RANDOM_SEED}-{MAX_RANDOM_SEED})，未能找到可成功求解的配置。")
        print(f"  测试时的客户端数量: {initial_num_clients}")
        print(f"  测试时的辅助节点数量: {initial_num_helpers}")

    try:
        with open(main_script_file_path, 'w', encoding='utf-8') as f:
            f.write(original_main_script_content)
        print(f"\n已将 {main_script_file_path} 恢复到原始内容。")
    except Exception as e:
        print(f"警告: 恢复 {main_script_file_path} 失败: {e}")
        print("请手动检查并恢复该文件。")