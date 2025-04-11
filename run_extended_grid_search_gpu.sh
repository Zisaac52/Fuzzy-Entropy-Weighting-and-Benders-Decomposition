#!/bin/bash

# --- Fixed Experiment Parameters ---
GPU_ID=0
NUM_NODES=8
METHOD="fuzzy"
# Learning rate 0.01 is default in main_node.py/main_server.py, no need to pass explicitly
# Dataset MNIST and IID True are defaults in run_gpu_experiment.sh, no need to pass explicitly

# --- Grid Search Parameters ---
M_VALUES=(2 3 5 7 9)
R_VALUES=(0.2 0.5 0.7 1.0 1.5)
EPOCH_VALUES=($(seq 1 10)) # Epochs from 1 to 10

# --- Expected Header for Results Files ---
EXPECTED_HEADER="Method,FuzzyM,FuzzyR,Nodes,Epochs,Accuracy,Loss,GPU_ID"
DEFAULT_RESULTS_FILE="gpu_experiment_results.csv" # Default file where run_gpu_experiment.sh writes

# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up background processes..."
    # Kill all background jobs spawned by this script's children (run_gpu_experiment.sh)
    pkill -P $$ # Kill direct children
    echo "Cleanup attempt complete."
}
trap cleanup EXIT SIGINT SIGTERM


# --- Run Grid Search for each Epoch ---
echo "Starting Extended Grid Search for Fuzzy Entropy parameters (m, r) across Epochs 1-10..."
echo "Fixed parameters: GPU=$GPU_ID, Nodes=$NUM_NODES, Method=$METHOD"

for current_epoch in "${EPOCH_VALUES[@]}"; do
    echo "====================================================="
    echo "Starting Grid Search for Epoch = $current_epoch"
    echo "====================================================="

    # --- Results File for the current epoch ---
    RESULTS_FILE="gpu_grid_search_results_node${NUM_NODES}_epoch${current_epoch}.csv"
    echo "Results for this epoch will be saved to: $RESULTS_FILE"

    # --- Initialize Results File Header for the current epoch ---
    if [ ! -f "$RESULTS_FILE" ] || ! grep -q "^$EXPECTED_HEADER" "$RESULTS_FILE"; then
        echo "Initializing results file: $RESULTS_FILE"
        echo "$EXPECTED_HEADER" > "$RESULTS_FILE"
    fi

    # --- Run Grid Search for m and r for the current epoch ---
    for m in "${M_VALUES[@]}"; do
        for r in "${R_VALUES[@]}"; do
            echo "-----------------------------------------------------"
            echo "Running experiment with: Epoch=$current_epoch, m=$m, r=$r"
            echo "-----------------------------------------------------"

            # Execute the single experiment script with the current epoch
            ./run_gpu_experiment.sh \
                --gpu $GPU_ID \
                --node $NUM_NODES \
                --epoch $current_epoch \
                --method $METHOD \
                --fuzzy_m $m \
                --fuzzy_r $r

            # Check the exit status
            if [ $? -ne 0 ]; then
                echo "Error running experiment for Epoch=$current_epoch, m=$m, r=$r. Check logs in ./logs/"
                # Append an error line to our specific grid search file
                 echo "$METHOD,$m,$r,$NUM_NODES,$current_epoch,RunError,RunError,$GPU_ID" >> "$RESULTS_FILE"
                # Continue to the next iteration
                continue
            fi

            # Extract the last line from the default results file and append it
            if [ -f "$DEFAULT_RESULTS_FILE" ]; then
                # Assume the last line is the correct one for this run
                tail -n 1 "$DEFAULT_RESULTS_FILE" >> "$RESULTS_FILE"
                echo "Result for Epoch=$current_epoch, m=$m, r=$r copied to $RESULTS_FILE"
            else
                 echo "Warning: Default results file $DEFAULT_RESULTS_FILE not found after running Epoch=$current_epoch, m=$m, r=$r."
                 # Append an error line to our grid search file
                 echo "$METHOD,$m,$r,$NUM_NODES,$current_epoch,CopyError,CopyError,$GPU_ID" >> "$RESULTS_FILE"
            fi

            echo "Finished experiment for Epoch=$current_epoch, m=$m, r=$r."
            sleep 5 # Small delay
        done # End r loop
    done # End m loop

    echo "====================================================="
    echo "Grid Search for Epoch = $current_epoch Completed."
    echo "Results saved in: $RESULTS_FILE"
    echo "====================================================="

done # End epoch loop

echo "-----------------------------------------------------"
echo "Extended Grid Search Completed for all epochs (1-10)."
echo "-----------------------------------------------------"

exit 0
