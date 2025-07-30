import subprocess
import re
import os
import csv
import datetime

# --- 配置 ---
MAIN_SCRIPT_MODULE = "gbd_resource_optimizer.main"
PYTHON_EXECUTABLE = "python" # Or your specific python interpreter, e.g., "python3.11"
SUMMARY_CSV_FILE = "gbd_summary_results.csv"
ITERATION_CSV_FILE = "gbd_iteration_details.csv"

def run_main_script_and_get_output():
    """执行主优化脚本并返回其标准输出"""
    try:
        # 假设此脚本与 gbd_resource_optimizer 目录在同一父目录下 (项目根目录)
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
        
        print(f"正在运行: {PYTHON_EXECUTABLE} -m {MAIN_SCRIPT_MODULE}")
        process = subprocess.run(
            [PYTHON_EXECUTABLE, "-m", MAIN_SCRIPT_MODULE],
            capture_output=True,
            text=True, # 获取字符串输出
            timeout=600, # 增加超时时间，例如10分钟
            cwd=project_root_dir,
            encoding='utf-8',
            errors='ignore'
        )
        print("主脚本运行完成。")
        # print("STDOUT:")
        # print(process.stdout)
        # if process.stderr:
        #     print("STDERR:")
        #     print(process.stderr)
        return process.stdout
    except subprocess.TimeoutExpired:
        print("主脚本运行超时。")
        return "ERROR: Timeout"
    except Exception as e:
        print(f"运行主脚本时发生错误: {e}")
        return f"ERROR: {e}"

def parse_overall_summary(log_content):
    """从日志内容中解析总体摘要信息"""
    summary = {}
    try:
        # Instance Info (从日志开头获取)
        match = re.search(r"INFO: Generated instance with (\d+) clients and (\d+) helpers.", log_content)
        if match:
            summary["Num_Clients"] = int(match.group(1))
            summary["Num_Helpers"] = int(match.group(2))
        
        # Heterogeneity and Seed (需要从特定日志行提取)
        # 假设 main.py 中 heterogeneity_level 和 effective_seed_for_this_run 会被 logging.info 打印
        match_het = re.search(r"INFO: Generated instance with .* heterogeneity: (\w+)\.", log_content)
        if match_het:
            summary["Heterogeneity_Level"] = match_het.group(1)
        
        match_seed = re.search(r"INFO: Run completed. Effective RANDOM_SEED for this run: (\S+)", log_content)
        if match_seed:
            seed_val = match_seed.group(1)
            summary["Seed"] = seed_val
            # 构建 Instance Name
            if "Num_Clients" in summary and "Num_Helpers" in summary and "Heterogeneity_Level" in summary:
                het_short = summary["Heterogeneity_Level"][:1] if summary["Heterogeneity_Level"] else "u"
                summary["Instance_Name_ID"] = f"c{summary['Num_Clients']}_h{summary['Num_Helpers']}_s{seed_val}_het{het_short}"
            else:
                summary["Instance_Name_ID"] = f"seed_{seed_val}_unknown_config"


        # 从 "GBD Algorithm Finished." 之后的部分提取
        summary_section = log_content.split("GBD Algorithm Finished.")[-1]
        
        match = re.search(r"Solution Status:\s*(.+)", summary_section)
        if match: summary["Solution_Status"] = match.group(1).strip()
        else: # 尝试从其他地方推断
            if "Converged successfully" in summary_section: summary["Solution_Status"] = "Converged"
            elif "Reached maximum iterations" in summary_section: summary["Solution_Status"] = "Max Iterations Reached"
            elif "Algorithm terminated prematurely." in summary_section: summary["Solution_Status"] = "Terminated Prematurely"
            elif "No feasible solution was found" in summary_section: summary["Solution_Status"] = "No Feasible Solution Found"


        match = re.search(r"Final Upper Bound \(Best UB Found\):\s*([-\w.]+)", summary_section)
        if match: summary["Final_Objective_Value"] = match.group(1).strip()
        
        match = re.search(r"Total GBD Time \(s\):\s*([\d\.]+)", summary_section)
        if match: summary["Total_GBD_Time_s"] = float(match.group(1))
            
        match = re.search(r"Total Iterations:\s*(\d+)", summary_section)
        if match: summary["Total_Iterations"] = int(match.group(1))

        match = re.search(r"Final Lower Bound \(LB\):\s*([-\w.]+)", summary_section)
        if match: summary["Final_LB"] = match.group(1).strip()

        match = re.search(r"Final Upper Bound \(Best UB Found\):\s*([-\w.]+)", summary_section) # 重复，与Objective一致
        if match: summary["Final_UB"] = match.group(1).strip()

        match = re.search(r"Final Gap:\s*([-\w.\s\(%\)]+)", summary_section) # 捕获包括百分比的gap字符串
        if match: summary["Final_Gap"] = match.group(1).strip()

        # Integer Solutions
        match_xi = re.search(r"Integer Variables \(x_i\):\s*(\{.*?\})", summary_section)
        if match_xi: summary["Optimal_x_i"] = match_xi.group(1)
        
        match_ni = re.search(r"Integer Variables \(N_i\):\s*(\{.*?\})", summary_section)
        if match_ni: summary["Optimal_N_i"] = match_ni.group(1)
        
        match_yij = re.search(r"Integer Variables \(Y_ij\):\s*(\{.*?\})", summary_section)
        if match_yij: summary["Optimal_Y_ij"] = match_yij.group(1)

        # Continuous Solutions (简化版，仅提取部分)
        # "p: {0: 0.999999976242482, 1: 0.6732287202524688}"
        cont_sol_text = ""
        p_match = re.search(r"p:\s*(\{.*?\})", summary_section)
        if p_match: cont_sol_text += f"p: {p_match.group(1)}; "
        
        fb_match = re.search(r"f_B:\s*(\{.*?\})", summary_section)
        if fb_match: cont_sol_text += f"f_B: {fb_match.group(1)}; "

        fl_match = re.search(r"f_L:\s*(\{.*?\})", summary_section)
        if fl_match: cont_sol_text += f"f_L: {fl_match.group(1)}; "
        
        fjr_match = re.search(r"f_j_r_assigned:\s*(\{.*?\})", summary_section)
        if fjr_match: cont_sol_text += f"f_j_r_assigned: {fjr_match.group(1)}; "
        summary["Optimal_Continuous_Solution"] = cont_sol_text.strip() if cont_sol_text else "N/A"
        
        # MP/SP/RSP times and counts
        match = re.search(r"Master Problem: Solved (\d+) times, Total Time = ([\d\.]+)s", summary_section)
        if match:
            summary["MP_Solve_Count"] = int(match.group(1))
            summary["MP_Total_Time_s"] = float(match.group(2))
        
        match = re.search(r"Subproblem \(SP\): Solved (\d+) times, Total Time = ([\d\.]+)s", summary_section)
        if match:
            summary["SP_Solve_Count"] = int(match.group(1))
            summary["SP_Total_Time_s"] = float(match.group(2))

        match = re.search(r"Relaxed Subproblem \(RSP\): Solved (\d+) times, Total Time = ([\d\.]+)s", summary_section)
        if match:
            summary["RSP_Solve_Count"] = int(match.group(1))
            summary["RSP_Total_Time_s"] = float(match.group(2))
        else:
            summary["RSP_Solve_Count"] = 0
            summary["RSP_Total_Time_s"] = 0.0
            
    except Exception as e:
        print(f"解析总体摘要时出错: {e}")
    return summary

def parse_iteration_data(log_content, instance_name):
    """从日志中解析每次迭代的 LB, UB, Gap"""
    iteration_logs = []
    # Pattern example: "End of Iteration 1: LB = -0.0822, UB = 0.7057"
    # Pattern example: "Current Gap: 0.7879 (111.66%)" or "Current Gap: inf"
    iteration_pattern = re.compile(
        r"--- Iteration (\d+) ---\n" # Start of iteration block (optional for context but not directly used)
        r".*?" # Non-greedy match for anything in between
        r"End of Iteration \1: LB = ([-\d\.]+), UB = ([-\d\.]+)\n"
        r"Current Gap:\s*([-\w\.\s\(%\)]+)", # Capture gap string
        re.DOTALL # Allow . to match newline
    )

    for match in iteration_pattern.finditer(log_content):
        try:
            iter_num = int(match.group(1))
            lb = float(match.group(2))
            ub = float(match.group(3))
            gap_str = match.group(4).strip()
            iteration_logs.append({
                "Instance_Name_ID": instance_name,
                "Iteration": iter_num,
                "LB": lb,
                "UB": ub,
                "Gap": gap_str
            })
        except Exception as e:
            print(f"解析迭代数据时出错 (迭代 {match.group(1) if match else 'unknown'}): {e}")
    return iteration_logs

def write_to_csv(data_list, csv_filepath, fieldnames):
    if not data_list:
        return
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        if isinstance(data_list, list):
            writer.writerows(data_list)
        else: # Single dictionary
            writer.writerow(data_list)
    print(f"数据已记录到 {csv_filepath}")

# --- 主逻辑 ---
if __name__ == "__main__":
    output_log = run_main_script_and_get_output()

    if "ERROR:" in output_log[:100]: # Check if script execution failed early
        print(f"主脚本运行失败，日志分析中止。错误: {output_log}")
        exit(1)

    summary_data = parse_overall_summary(output_log)
    
    if not summary_data.get("Instance_Name_ID"):
        summary_data["Instance_Name_ID"] = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"警告: 未能从日志生成完整的Instance_Name_ID, 使用默认值: {summary_data['Instance_Name_ID']}")

    # 定义summary CSV的表头顺序
    summary_fieldnames = [
        "Timestamp", "Instance_Name_ID", "Num_Clients", "Num_Helpers", "Heterogeneity_Level", "Seed",
        "Solution_Status", "Final_Objective_Value", "Total_GBD_Time_s", "Total_Iterations", 
        "Final_LB", "Final_UB", "Final_Gap", 
        "MP_Solve_Count", "MP_Total_Time_s", "SP_Solve_Count", "SP_Total_Time_s", "RSP_Solve_Count", "RSP_Total_Time_s",
        "Optimal_x_i", "Optimal_N_i", "Optimal_Y_ij", "Optimal_Continuous_Solution"
    ]
    # 添加时间戳
    summary_data["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 确保所有表头字段都在summary_data中，如果缺少则设为N/A
    for field in summary_fieldnames:
        if field not in summary_data:
            summary_data[field] = "N/A"
            
    write_to_csv([summary_data], SUMMARY_CSV_FILE, summary_fieldnames)

    iteration_details = parse_iteration_data(output_log, summary_data["Instance_Name_ID"])
    if iteration_details:
        iteration_fieldnames = ["Instance_Name_ID", "Iteration", "LB", "UB", "Gap"]
        write_to_csv(iteration_details, ITERATION_CSV_FILE, iteration_fieldnames)
    
    print("\n日志解析和记录完成。")