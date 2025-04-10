#!/bin/bash

# --- Fixed Experiment Parameters ---
GPU_ID=0
NUM_NODES=8
EPOCHS=2 # Changed from 5 to 2
METHOD="fuzzy"
# Learning rate 0.01 is default in main_node.py/main_server.py, no need to pass explicitly
# Dataset MNIST and IID True are defaults in run_gpu_experiment.sh, no need to pass explicitly

# --- Grid Search Parameters ---
M_VALUES=(2 3 5 7 9)
R_VALUES=(0.2 0.5 0.7 1.0 1.5)

# --- Results File ---
# Use a dedicated results file for this grid search to avoid mixing with other results
RESULTS_FILE="gpu_grid_search_results_node${NUM_NODES}_epoch${EPOCHS}.csv" # Filename will now reflect epoch=2
EXPECTED_HEADER="Method,FuzzyM,FuzzyR,Nodes,Epochs,Accuracy,Loss,GPU_ID"

# --- Initialize Results File Header ---
if [ ! -f "$RESULTS_FILE" ] || ! grep -q "^$EXPECTED_HEADER" "$RESULTS_FILE"; then
    echo "Initializing results file: $RESULTS_FILE"
    echo "$EXPECTED_HEADER" > "$RESULTS_FILE"
fi

# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up background processes..."
    # Kill all background jobs spawned by this script's children (run_gpu_experiment.sh)
    # This is tricky, might need refinement if processes linger.
    # A simpler approach is to let run_gpu_experiment.sh handle its own cleanup via trap.
    pkill -P $$ # Kill direct children (the run_gpu_experiment.sh instances)
    echo "Cleanup attempt complete. Individual experiment scripts should handle deeper cleanup."
}
trap cleanup EXIT SIGINT SIGTERM


# --- Run Grid Search ---
echo "Starting Grid Search for Fuzzy Entropy parameters (m, r)..."
echo "Fixed parameters: GPU=$GPU_ID, Nodes=$NUM_NODES, Epochs=$EPOCHS, Method=$METHOD" # Will now print Epochs=2
echo "Results will be saved to: $RESULTS_FILE" # Will now print the epoch=2 filename

for m in "${M_VALUES[@]}"; do
    for r in "${R_VALUES[@]}"; do
        echo "-----------------------------------------------------"
        echo "Running experiment with: m=$m, r=$r"
        echo "-----------------------------------------------------"

        # Execute the single experiment script
        # It will append its result to gpu_experiment_results.csv by default.
        # We rely on run_gpu_experiment.sh to handle server/node start/stop and result extraction.
        ./run_gpu_experiment.sh \
            --gpu $GPU_ID \
            --node $NUM_NODES \
            --epoch $EPOCHS \
            --method $METHOD \
            --fuzzy_m $m \
            --fuzzy_r $r

        # Check the exit status of the last command
        if [ $? -ne 0 ]; then
            echo "Error running experiment for m=$m, r=$r. Check logs in ./logs/"
            # Optionally decide whether to continue or exit the grid search on error
            # continue
            # exit 1
        fi

        # Extract the last line from the default results file and append it to our specific grid search file
        # This assumes run_gpu_experiment.sh successfully wrote to its default file
        DEFAULT_RESULTS_FILE="gpu_experiment_results.csv"
        if [ -f "$DEFAULT_RESULTS_FILE" ]; then
            # Get the last line that matches the current parameters (best effort)
            # This is slightly fragile if multiple runs have identical parameters
            # A safer approach would be modifying run_gpu_experiment.sh to accept an output file argument.
            # For now, let's assume the last line is the correct one.
            tail -n 1 "$DEFAULT_RESULTS_FILE" >> "$RESULTS_FILE"
            echo "Result for m=$m, r=$r copied to $RESULTS_FILE"
        else
             echo "Warning: Default results file $DEFAULT_RESULTS_FILE not found after running m=$m, r=$r."
             # Append an error line to our grid search file
             echo "$METHOD,$m,$r,$NUM_NODES,$EPOCHS,CopyError,CopyError,$GPU_ID" >> "$RESULTS_FILE"
        fi

        echo "Finished experiment for m=$m, r=$r."
        sleep 5 # Add a small delay between experiments if needed
    done
done

echo "-----------------------------------------------------"
echo "Grid Search Completed."
echo "Results saved in: $RESULTS_FILE" # Filename reflects node 8
echo "-----------------------------------------------------"

exit 0
