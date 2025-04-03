#!/bin/bash

# Script to run experiments for a fixed number of nodes (8)
# iterating through epochs 1 to 10 and methods fuzzy, iewm, fedasync.

NUM_NODES=8
METHODS=("fuzzy" "iewm" "fedasync")
EPOCHS=($(seq 1 10)) # Create a sequence from 1 to 10

# Ensure the script is executable
chmod +x ./run_experiment.sh

echo "Starting batch experiments for Node=${NUM_NODES}"
echo "Epochs: 1 to 10"
echo "Methods: ${METHODS[@]}"
echo "--------------------------------------------------"

# Loop through each method
for method in "${METHODS[@]}"; do
  # Loop through each epoch
  for epoch in "${EPOCHS[@]}"; do
    echo ""
    echo ">>> Running experiment: Method=${method}, Nodes=${NUM_NODES}, Epochs=${epoch} <<<"
    echo ""

    # Run the individual experiment script
    ./run_experiment.sh --node=${NUM_NODES} --epoch=${epoch} --method=${method}

    # Optional: Add a small delay between experiments if needed
    # sleep 5
  done
done

echo "--------------------------------------------------"
echo "All experiments for Node=${NUM_NODES} completed."
