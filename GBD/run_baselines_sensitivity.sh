#!/bin/bash

# 定义要测试的参数
WORKLOADS=(100 200 300 400 500)
BASELINES=("local_first" "greedy_offload" "random_offload" "all_offload")
RUNS_PER_CONFIG=10
LOG_FILE="baselines_sensitivity_log.csv"
NUM_CLIENTS=20
NUM_HELPERS=2

# 循环遍历所有配置并运行实验
echo "Starting baseline sensitivity analysis..."
echo "Results will be logged to $LOG_FILE"
echo "---"

for workload in "${WORKLOADS[@]}"; do
    for baseline in "${BASELINES[@]}"; do
        echo "Running configuration: Workload=$workload, Solver=$baseline, Runs=$RUNS_PER_CONFIG"
        for (( run=1; run<=$RUNS_PER_CONFIG; run++ )); do
            echo "  -> Run $run/$RUNS_PER_CONFIG"
            python3 run_experiment.py \
                --task_workload "$workload" \
                --solver "$baseline" \
                --num_clients $NUM_CLIENTS \
                --num_helpers $NUM_HELPERS \
                --log_file "$LOG_FILE"
        done
        echo "Configuration completed."
        echo "---"
    done
done

echo "All baseline sensitivity experiments completed."
echo "Results are in $LOG_FILE"