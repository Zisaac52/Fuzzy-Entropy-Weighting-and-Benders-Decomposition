#!/bin/bash

# Default values
NUM_NODES=4
EPOCHS=1
METHOD="fuzzy"
FUZZY_M=2 # Default fuzzy_m value
GPU_ID=-1 # Default to CPU, must be overridden by user

# --- Parse Command Line Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --node) NUM_NODES="$2"; shift ;;
        --epoch) EPOCHS="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --fuzzy_m) FUZZY_M="$2"; shift ;;
        --gpu) GPU_ID="$2"; shift ;; # Mandatory GPU ID
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate GPU ID ---
if [ "$GPU_ID" -eq -1 ]; then
    echo "Error: --gpu parameter is required and must be a non-negative integer (e.g., --gpu 0)."
    exit 1
fi
echo "Running experiment on GPU: $GPU_ID"

# --- Validate Method ---
if [[ "$METHOD" != "fuzzy" && "$METHOD" != "iewm" && "$METHOD" != "fedasync" ]]; then
    echo "Error: Invalid method specified. Choose 'fuzzy', 'iewm', or 'fedasync'."
    exit 1
fi

# --- Setup Log Directory ---
LOG_DIR="logs_gpu${GPU_ID}_method_${METHOD}_node_${NUM_NODES}_epoch_${EPOCHS}"
if [ "$METHOD" == "fuzzy" ]; then
    LOG_DIR="${LOG_DIR}_fuzzym${FUZZY_M}"
fi
mkdir -p "$LOG_DIR"
echo "Logs will be saved in: $LOG_DIR"

# --- Setup Results File ---
RESULTS_FILE="gpu_experiment_results.csv"
echo "GPU experiment results will be appended to: $RESULTS_FILE"

# --- Server Port ---
SERVER_PORT=8000 # Keep consistent with node default

# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up background processes..."
    # Kill all background jobs spawned by this script
    kill $(jobs -p) 2>/dev/null
    wait # Wait for processes to terminate
    echo "Cleanup complete."
}
trap cleanup EXIT SIGINT SIGTERM

# --- Start Server ---
echo "Starting server on port $SERVER_PORT using GPU $GPU_ID..."
python main_server.py \
    --port $SERVER_PORT \
    --aggregate $METHOD \
    --fuzzy_m $FUZZY_M \
    --gpu $GPU_ID \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
sleep 5 # Give server time to start

# --- Start Nodes ---
TOTAL_TRAIN_DATA=60000 # MNIST total training samples
TOTAL_TEST_DATA=10000  # MNIST total test samples
TRAIN_SLICE=$((TOTAL_TRAIN_DATA / NUM_NODES))
TEST_SLICE=$((TOTAL_TEST_DATA / NUM_NODES))

echo "Starting $NUM_NODES nodes..."
for i in $(seq 0 $((NUM_NODES - 1)))
do
    START_TRAIN=$((i * TRAIN_SLICE))
    END_TRAIN=$(((i + 1) * TRAIN_SLICE))
    START_TEST=$((i * TEST_SLICE))
    END_TEST=$(((i + 1) * TEST_SLICE))
    NODE_ADDRESS="node_$i" # Simple address for identification

    echo "Starting Node $i (Address: $NODE_ADDRESS) on GPU $GPU_ID..."
    python main_node.py \
        --epoch $EPOCHS \
        --gpu $GPU_ID \
        --dataset mnist \
        --iid true \
        --lr 0.001 \
        --port $SERVER_PORT \
        --address $NODE_ADDRESS \
        --start_train_index $START_TRAIN \
        --end_train_index $END_TRAIN \
        --start_test_index $START_TEST \
        --end_test_index $END_TEST \
        --user_id $i \
        > "$LOG_DIR/node_$i.log" 2>&1 &
    NODE_PIDS+=($!) # Store node PIDs if needed later, though trap handles cleanup
    sleep 1 # Stagger node starts slightly
done

echo "All nodes started. Waiting for completion..."

# --- Wait for all node background jobs to finish ---
# We wait for node PIDs specifically, not the server
if [ ${#NODE_PIDS[@]} -ne 0 ]; then
    wait ${NODE_PIDS[@]}
fi
echo "All nodes finished."

# --- Evaluate Final Model (Server should still be running) ---
echo "Evaluating final model on GPU $GPU_ID (fetching from server)..."
# Run evaluation script and capture output directly
# Give the server a brief moment just in case aggregation needs finalization (optional, can be removed if not needed)
sleep 2
EVAL_OUTPUT=$(python view_aggregated_model.py --port $SERVER_PORT --gpu $GPU_ID 2>&1)
echo "--- Evaluation Script Output Start ---"
echo "$EVAL_OUTPUT"
echo "--- Evaluation Script Output End ---"

# Extract accuracy - Looking for "Global Model Test: Accuracy: XX.XX%"
FINAL_ACC=$(echo "$EVAL_OUTPUT" | grep -oP 'Global Model Test: Accuracy: \K[0-9]+\.[0-9]+')
# Extract loss - Looking for "Average Loss: Y.YYYY"
FINAL_LOSS=$(echo "$EVAL_OUTPUT" | grep -oP 'Average Loss: \K[0-9]+\.[0-9]+')


# --- Stop the Server (Now that evaluation is done) ---
echo "Stopping server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null # Wait for server to terminate, ignore errors if already stopped
echo "Server stopped."

# --- Record Results ---
# Define FuzzyM value for output, default to N/A
FUZZY_M_VAL="N/A"
if [ "$METHOD" == "fuzzy" ]; then
    FUZZY_M_VAL=$FUZZY_M
fi

# Check and initialize results file header if needed
EXPECTED_HEADER="Method,FuzzyM,Nodes,Epochs,Accuracy,Loss,GPU_ID" # Added Loss column
INITIALIZE_HEADER=false
if [ -f "$RESULTS_FILE" ]; then
    CURRENT_HEADER=$(head -n 1 "$RESULTS_FILE")
    if [ "$CURRENT_HEADER" != "$EXPECTED_HEADER" ]; then
        echo "Warning: Results file header mismatch in $RESULTS_FILE."
        BACKUP_FILE="${RESULTS_FILE}.$(date +%Y%m%d_%H%M%S).bak"
        echo "Backing up existing file to $BACKUP_FILE"
        mv "$RESULTS_FILE" "$BACKUP_FILE"
        INITIALIZE_HEADER=true
    fi
else
    INITIALIZE_HEADER=true
fi

if [ "$INITIALIZE_HEADER" = true ]; then
    echo "Initializing results file: $RESULTS_FILE with header: $EXPECTED_HEADER"
    echo "$EXPECTED_HEADER" > "$RESULTS_FILE"
fi

# Append result to the CSV file
# Check if both accuracy and loss were extracted successfully
if [ -n "$FINAL_ACC" ] && [ -n "$FINAL_LOSS" ]; then
    # Accuracy and Loss were extracted successfully
    echo "$METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,$FINAL_ACC,$FINAL_LOSS,$GPU_ID" >> "$RESULTS_FILE"
    echo "Result appended to $RESULTS_FILE"
elif [ -n "$FINAL_ACC" ]; then
    # Only Accuracy was extracted
     echo "$METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,$FINAL_ACC,LossError,$GPU_ID" >> "$RESULTS_FILE"
     echo "Accuracy recorded, but failed to extract Loss. Result appended to $RESULTS_FILE"
else
    # Accuracy extraction failed (implies loss extraction might also fail or be irrelevant)
    echo "Could not extract accuracy (and possibly loss) from evaluation output. Check evaluation output above."
    echo "$METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,EvalError,EvalError,$GPU_ID" >> "$RESULTS_FILE"
    echo "Evaluation error appended to $RESULTS_FILE"
fi


# --- Display Final Result Clearly ---
if [ -n "$FINAL_ACC" ] && [ -n "$FINAL_LOSS" ]; then
    echo "-----------------------------------------------------"
    echo "Final Global Model Accuracy: $FINAL_ACC%"
    echo "Final Global Model Loss: $FINAL_LOSS"
    echo "-----------------------------------------------------"
elif [ -n "$FINAL_ACC" ]; then
     echo "-----------------------------------------------------"
     echo "Final Global Model Accuracy: $FINAL_ACC%"
     echo "Final Global Model Loss: Extraction Error"
     echo "-----------------------------------------------------"
else
    echo "-----------------------------------------------------"
    echo "Final Global Model Accuracy: Evaluation Error"
    echo "Final Global Model Loss: Evaluation Error"
    echo "-----------------------------------------------------"
fi

echo "Experiment script finished."
exit 0
