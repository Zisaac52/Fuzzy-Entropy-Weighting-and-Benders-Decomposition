#!/bin/bash

# ==============================================================================
# Script for running task workload sensitivity analysis experiments.
#
# This script iterates over a predefined set of task workloads. For each
# workload, it runs the GBD experiment until 10 successful (converged)
# results are collected. All baseline comparisons are also run for each attempt.
#
# Results are logged to a single CSV file.
#
# Usage:
#   ./run_workload_sensitivity.sh
# ==============================================================================

# --- Configuration ---
# Array of task workloads to test (in KB)
WORKLOADS=(100 200 300 400 500)
REQUIRED_SUCCESSFUL_RUNS=10

# Fixed parameters for the experiment
NUM_CLIENTS=20
NUM_HELPERS=2
SOLVER="all"
LOG_FILE="workload_sensitivity_log.csv"

# Get the directory of this script to ensure python is called from the correct path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# --- Main Experiment Loop ---
echo "Starting workload sensitivity analysis..."
echo "Will collect ${REQUIRED_SUCCESSFUL_RUNS} converged GBD results for each workload."
echo "Results will be logged to: ${LOG_FILE}"

# Remove the log file if it exists to start fresh for this experiment batch
if [ -f "${SCRIPT_DIR}/${LOG_FILE}" ]; then
    rm "${SCRIPT_DIR}/${LOG_FILE}"
    echo "Removed existing log file: ${LOG_FILE}"
fi

# Loop over each workload value
for workload in "${WORKLOADS[@]}"
do
    echo ""
    echo "=============================================================================="
    echo "Processing Workload = ${workload} KB"
    echo "=============================================================================="

    # Loop to collect the required number of successful runs for the current workload
    for run in $(seq 1 ${REQUIRED_SUCCESSFUL_RUNS})
    do
        echo ""
        echo "--- [Workload: ${workload} KB] | Starting Run ${run}/${REQUIRED_SUCCESSFUL_RUNS} ---"
        
        successful_seed=""
        find_seed_attempts=0

        # Stage 1: Find a successful seed that allows GBD to converge
        while true
        do
            find_seed_attempts=$((find_seed_attempts + 1))
            # Generate a random seed for this attempt
            seed=$(od -A n -t d -N 4 /dev/urandom | tr -d ' ')

            echo "Finding seed (Attempt ${find_seed_attempts})... Trying seed: ${seed}"
            
            # Run GBD ONLY to check for convergence. Log to a temporary, throwaway file.
            python3 "${SCRIPT_DIR}/run_experiment.py" \
                --solver gbd \
                --num_clients ${NUM_CLIENTS} \
                --num_helpers ${NUM_HELPERS} \
                --task_workload ${workload} \
                --seed ${seed} \
                --log_file "temp_log.csv"

            # Check exit code
            if [ $? -eq 0 ]; then
                successful_seed=${seed}
                echo "Found a successful seed: ${successful_seed}"
                rm temp_log.csv # Clean up the temporary log file
                break # Exit the while loop
            else
                echo "Seed ${seed} did not converge for GBD. Trying another..."
            fi

            # Safety break
            if [ ${find_seed_attempts} -gt 100 ]; then
                echo "Error: Failed to find a converging seed after 100 attempts for workload ${workload}. Aborting."
                exit 1
            fi
        done

        # Stage 2: Run all solvers with the successful seed
        if [ -n "${successful_seed}" ]; then
            # Generate a unique run_id for this batch of solver runs
            run_id="workload${workload}_run${run}"
            echo "Executing all solvers with successful_seed=${successful_seed} and run_id=${run_id}"

            python3 "${SCRIPT_DIR}/run_experiment.py" \
                --solver ${SOLVER} \
                --num_clients ${NUM_CLIENTS} \
                --num_helpers ${NUM_HELPERS} \
                --task_workload ${workload} \
                --seed ${successful_seed} \
                --run_id ${run_id} \
                --log_file ${LOG_FILE}
            
            echo "--- Finished Run ${run}/${REQUIRED_SUCCESSFUL_RUNS} for Workload ${workload} KB ---"
        else
            echo "Could not find a successful seed for workload ${workload}. Skipping this run."
        fi
    done
    echo "Successfully completed all runs for workload ${workload} KB."
done

echo ""
echo "=============================================================================="
echo "Workload sensitivity analysis finished for all workloads."
echo "All results have been saved to ${LOG_FILE}"
echo "=============================================================================="