#!/bin/bash

# Script to run GPU experiments for a fixed number of nodes (8)
# iterating through epochs 1 to 10 and methods fuzzy, iewm, fedasync.

NUM_NODES=8
GPU_ID=0 # Use GPU 0 for these experiments
METHODS=("fuzzy" "iewm" "fedasync") # all experiment methods
# METHODS=("iewm" "fedasync") # Exclude fuzzy for this script
EPOCHS=($(seq 1 10)) # Create a sequence from 1 to 10
FUZZY_M_VALUES=(5) # Define fuzzy_m values to test

# Ensure the target script is executable
chmod +x ./run_gpu_experiment.sh

echo "Starting batch GPU experiments for Node=${NUM_NODES} on GPU=${GPU_ID}"
echo "Epochs: 1 to 10"
echo "Methods: ${METHODS[@]}"
echo "Fuzzy M values (for fuzzy method): ${FUZZY_M_VALUES[@]}"
echo "--------------------------------------------------"

# Loop through each method
for method in "${METHODS[@]}"; do
  # Loop through each epoch
  for epoch in "${EPOCHS[@]}"; do
    if [ "$method" == "fuzzy" ]; then
      # Loop through fuzzy_m values only if method is fuzzy
      for m in "${FUZZY_M_VALUES[@]}"; do
        echo ""
        echo ">>> Running GPU experiment: Method=${method}, FuzzyM=${m}, Nodes=${NUM_NODES}, Epochs=${epoch}, GPU=${GPU_ID} <<<"
        echo ""
        # Run the individual GPU experiment script with fuzzy_m and gpu id (using space separator)
        ./run_gpu_experiment.sh --node ${NUM_NODES} --epoch ${epoch} --method ${method} --fuzzy_m ${m} --gpu ${GPU_ID}
        # Optional: Add a small delay between experiments if needed
        # sleep 5
      done
    else
      # Run non-fuzzy methods without fuzzy_m loop
      echo ""
      echo ">>> Running GPU experiment: Method=${method}, Nodes=${NUM_NODES}, Epochs=${epoch}, GPU=${GPU_ID} <<<"
      echo ""
      # Run the individual GPU experiment script without fuzzy_m but with gpu id (using space separator)
      ./run_gpu_experiment.sh --node ${NUM_NODES} --epoch ${epoch} --method ${method} --gpu ${GPU_ID}
      # Optional: Add a small delay between experiments if needed
      # sleep 5
    fi
  done
done

echo "--------------------------------------------------"
echo "All GPU experiments for Node=${NUM_NODES} on GPU=${GPU_ID} completed."
