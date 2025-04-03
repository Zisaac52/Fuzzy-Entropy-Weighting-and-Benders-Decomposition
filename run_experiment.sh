#!/bin/bash

# --- Default Configuration ---
DEFAULT_NUM_NODES=4
DEFAULT_EPOCHS=1
DEFAULT_AGGREGATE_METHOD="fuzzy" # Default method
DEFAULT_FUZZY_M=2 # Default fuzzy_m value
SERVER_PORT=8080
# AGGREGATE_METHOD will be set by args
DATASET="mnist"
LEARNING_RATE=0.01
TOTAL_TRAIN_SAMPLES=60000
TOTAL_TEST_SAMPLES=10000
PYTHON_CMD="python3" # Or just "python" if that's your command
RESULTS_FILE="experiment_results.csv"

# --- Argument Parsing ---
NUM_NODES=$DEFAULT_NUM_NODES
EPOCHS=$DEFAULT_EPOCHS
AGGREGATE_METHOD=$DEFAULT_AGGREGATE_METHOD
FUZZY_M=$DEFAULT_FUZZY_M # Initialize fuzzy_m

# Use getopt to parse named arguments
# Add --fuzzy_m to getopt
TEMP=$(getopt -o '' --long node:,epoch:,method:,fuzzy_m: -n 'run_experiment.sh' -- "$@")
if [ $? != 0 ] ; then echo "Argument parsing error. Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP" # Correctly quote the arguments for eval

while true; do
  case "$1" in
    --node ) NUM_NODES="$2"; shift 2 ;;
    --epoch ) EPOCHS="$2"; shift 2 ;;
    --method ) AGGREGATE_METHOD="$2"; shift 2 ;;
    --fuzzy_m ) FUZZY_M="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) echo "Internal error!" ; exit 1 ;; # Should not happen with getopt
  esac
done

# Basic validation
if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]] || ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "Error: --node and --epoch must be positive integers." >&2
    exit 1
fi
# Validate method
case "$AGGREGATE_METHOD" in
    fuzzy|iewm|fedasync) ;; # Allowed methods
    *) echo "Error: Invalid --method specified. Use 'fuzzy', 'iewm', or 'fedasync'." >&2 ; exit 1 ;;
esac
# Validate fuzzy_m if method is fuzzy
if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then
    if ! [[ "$FUZZY_M" =~ ^[0-9]+$ ]] || [ "$FUZZY_M" -le 0 ]; then
        echo "Error: --fuzzy_m must be a positive integer when method is 'fuzzy'." >&2
        exit 1
    fi
fi

# Update echo statement to include fuzzy_m if applicable
if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then
    echo "Running experiment with Method: $AGGREGATE_METHOD, FuzzyM: $FUZZY_M, Nodes: $NUM_NODES, Epochs: $EPOCHS..."
else
    echo "Running experiment with Method: $AGGREGATE_METHOD, Nodes: $NUM_NODES, Epochs: $EPOCHS..."
fi

# --- Ensure Logs Directory Exists ---
mkdir -p logs # Ensure the main logs directory exists

# --- Initialize Results File ---
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Initializing results file: $RESULTS_FILE"
    # Add Method and FuzzyM columns
    echo "Method,FuzzyM,Nodes,Epochs,Accuracy" > "$RESULTS_FILE"
fi

# --- Start Server ---
# Ensure the main logs directory exists
mkdir -p logs
# Construct server command, adding fuzzy_m if method is fuzzy
SERVER_CMD="$PYTHON_CMD ./main_server.py --port $SERVER_PORT --aggregate $AGGREGATE_METHOD"
if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then
    SERVER_CMD="$SERVER_CMD --fuzzy_m $FUZZY_M"
    echo "Starting server on port $SERVER_PORT with aggregation method '$AGGREGATE_METHOD' and fuzzy_m '$FUZZY_M'..."
else
    echo "Starting server on port $SERVER_PORT with aggregation method '$AGGREGATE_METHOD'..."
fi
# Redirect server output to the main logs directory, overwriting previous log
$SERVER_CMD > "logs/server.log" 2>&1 &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID. Log file: logs/server.log. Waiting a few seconds for it to initialize..."
sleep 5 # Give server time to start

# --- Check if server started successfully (basic check) ---
if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server failed to start. Check logs/server.log for details."
    # Record failure in results file, include FuzzyM
    FUZZY_M_VAL="N/A"
    if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then FUZZY_M_VAL=$FUZZY_M; fi
    echo "$AGGREGATE_METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,ServerError" >> "$RESULTS_FILE"
    exit 1
fi
echo "Server seems to be running."

# --- Start Nodes ---
NODE_PIDS=()
echo "Starting $NUM_NODES nodes..."

# Calculate samples per node
TRAIN_SAMPLES_PER_NODE=$((TOTAL_TRAIN_SAMPLES / NUM_NODES))
TEST_SAMPLES_PER_NODE=$((TOTAL_TEST_SAMPLES / NUM_NODES))

for i in $(seq 0 $((NUM_NODES - 1)))
do
    USER_ID=$i
    ADDRESS="Node${i}" # Simple address generation

    # Calculate data indices for this node
    START_TRAIN_INDEX=$((i * TRAIN_SAMPLES_PER_NODE))
    END_TRAIN_INDEX=$(((i + 1) * TRAIN_SAMPLES_PER_NODE))
    START_TEST_INDEX=$((i * TEST_SAMPLES_PER_NODE))
    END_TEST_INDEX=$(((i + 1) * TEST_SAMPLES_PER_NODE))

    # Adjust indices for the last node to include remaining samples
    if [ $i -eq $((NUM_NODES - 1)) ]; then
        END_TRAIN_INDEX=$TOTAL_TRAIN_SAMPLES
        END_TEST_INDEX=$TOTAL_TEST_SAMPLES
    fi

    echo "Starting Node $USER_ID (Address: $ADDRESS) with Epochs: $EPOCHS, Train Data [$START_TRAIN_INDEX:$END_TRAIN_INDEX], Test Data [$START_TEST_INDEX:$END_TEST_INDEX]"

    # Construct node command - Use the parsed EPOCHS value
    NODE_CMD="$PYTHON_CMD ./main_node.py \
        --user_id=$USER_ID \
        --iid=1 \
        --epoch=$EPOCHS \
        --address=$ADDRESS \
        --port=$SERVER_PORT \
        --lr=$LEARNING_RATE \
        --dataset=$DATASET \
        --start_train_index=$START_TRAIN_INDEX \
        --end_train_index=$END_TRAIN_INDEX \
        --start_test_index=$START_TEST_INDEX \
        --end_test_index=$END_TEST_INDEX"

    # Define node log file path, include fuzzy_m if method is fuzzy
    if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then
        NODE_LOG_FILE="logs/node_${USER_ID}_${AGGREGATE_METHOD}_m${FUZZY_M}_n${NUM_NODES}_e${EPOCHS}.log"
    else
        NODE_LOG_FILE="logs/node_${USER_ID}_${AGGREGATE_METHOD}_n${NUM_NODES}_e${EPOCHS}.log"
    fi
    echo "Node log file: $NODE_LOG_FILE"
    # Run node in background and log output to the main logs directory
    $NODE_CMD > "$NODE_LOG_FILE" 2>&1 &
    NODE_PIDS+=($!) # Store PID
done

# --- Wait for Nodes ---
echo "Waiting for all nodes to complete..."
EXIT_STATUS=0
for pid in "${NODE_PIDS[@]}"; do
    wait $pid
    # Capture exit status of nodes; if any failed, record it
    if [ $? -ne 0 ]; then
        EXIT_STATUS=1
        # Update error message to reflect new log location/naming
        if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then
            echo "Node with PID $pid failed. Check logs/node_${USER_ID}_${AGGREGATE_METHOD}_m${FUZZY_M}_n${NUM_NODES}_e${EPOCHS}.log for details."
        else
             echo "Node with PID $pid failed. Check logs/node_${USER_ID}_${AGGREGATE_METHOD}_n${NUM_NODES}_e${EPOCHS}.log for details."
        fi
    fi
done

if [ $EXIT_STATUS -ne 0 ]; then
    echo "One or more nodes failed to complete successfully."
    # Record failure, include FuzzyM
    FUZZY_M_VAL="N/A"
    if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then FUZZY_M_VAL=$FUZZY_M; fi
    echo "$AGGREGATE_METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,NodeError" >> "$RESULTS_FILE"
    # Stop server and exit
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID
    sleep 2
    kill -9 $SERVER_PID 2>/dev/null || true # Force kill if needed
    exit 1
fi

echo "All nodes have completed."

# --- Evaluate Final Model and Record Accuracy ---
echo "Evaluating the final aggregated model..."
# Run evaluation script and capture its output
EVAL_OUTPUT=$($PYTHON_CMD ./view_aggregated_model.py --port $SERVER_PORT)
echo "--- Evaluation Script Output Start ---"
echo "$EVAL_OUTPUT"
echo "--- Evaluation Script Output End ---"

# Extract accuracy - Looking for "Global Model Test: Accuracy: XX.XX%"
ACCURACY=$(echo "$EVAL_OUTPUT" | grep -oP 'Global Model Test: Accuracy: \K[0-9]+\.[0-9]+')

if [ -n "$ACCURACY" ]; then
    echo "Extracted Accuracy: $ACCURACY%"
    # Append result to CSV file including the method and FuzzyM
    FUZZY_M_VAL="N/A"
    if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then FUZZY_M_VAL=$FUZZY_M; fi
    echo "$AGGREGATE_METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,$ACCURACY" >> "$RESULTS_FILE"
    echo "Result recorded in $RESULTS_FILE"
else
    echo "Could not extract accuracy from evaluation output. Check logs/node_* logs and evaluation script."
    # Record failure or placeholder, include FuzzyM
    FUZZY_M_VAL="N/A"
    if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then FUZZY_M_VAL=$FUZZY_M; fi
    echo "$AGGREGATE_METHOD,$FUZZY_M_VAL,$NUM_NODES,$EPOCHS,EvalError" >> "$RESULTS_FILE"
fi


# --- Stop Server ---
echo "Stopping server (PID $SERVER_PID)..."
kill $SERVER_PID
# Wait a moment to ensure the process is killed
sleep 2
if ps -p $SERVER_PID > /dev/null; then
   echo "Server did not stop gracefully, sending SIGKILL..."
   kill -9 $SERVER_PID
fi
echo "Server stopped."

# Update final message
if [ "$AGGREGATE_METHOD" == "fuzzy" ]; then
    echo "Experiment finished for Method=$AGGREGATE_METHOD, FuzzyM=$FUZZY_M, Node=$NUM_NODES, Epoch=$EPOCHS."
else
    echo "Experiment finished for Method=$AGGREGATE_METHOD, Node=$NUM_NODES, Epoch=$EPOCHS."
fi
