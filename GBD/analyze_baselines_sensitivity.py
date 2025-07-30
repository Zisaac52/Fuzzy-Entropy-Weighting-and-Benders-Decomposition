import pandas as pd
import collections

def analyze_baselines_sensitivity(log_file):
    """
    Analyzes the baselines sensitivity log file to compute average performance metrics.

    Args:
        log_file (str): The path to the log file.
    """
    # Load the data from the CSV file
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: The file {log_file} was not found.")
        return

    # Group by solver and task_workload
    grouped = df.groupby(['solver', 'task_workload'])

    # Calculate the mean for total_time and total_energy
    mean_metrics = grouped[['total_time', 'total_energy']].mean()

    # Create a nested dictionary to store the results
    results = collections.defaultdict(dict)
    for index, row in mean_metrics.iterrows():
        solver, task_workload = index
        results[solver][task_workload] = {
            'avg_total_time': row['total_time'],
            'avg_total_energy': row['total_energy']
        }

    # Print the results in a structured format
    print("="*80)
    print("Baseline Strategies Sensitivity Analysis Results")
    print("="*80)
    for solver, workloads in sorted(results.items()):
        print(f"\n--- Strategy: {solver} ---")
        sorted_workloads = sorted(workloads.items())
        for workload, metrics in sorted_workloads:
            print(f"  Task Workload: {workload}")
            print(f"    - Average Total Time:   {metrics['avg_total_time']:.4f}")
            print(f"    - Average Total Energy: {metrics['avg_total_energy']:.4f}")
    print("\n" + "="*80)


if __name__ == "__main__":
    log_file_path = 'baselines_sensitivity_log.csv'
    analyze_baselines_sensitivity(log_file_path)