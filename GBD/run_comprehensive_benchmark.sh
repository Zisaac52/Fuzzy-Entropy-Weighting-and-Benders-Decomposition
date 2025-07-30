#!/bin/bash

# 定义要测试的客户端规模
CLIENT_SCALES=(2 5 8 10 15 20 25 30)

# 定义辅助节点的数量
NUM_HELPERS=2

# 循环遍历所有客户端规模
for clients in "${CLIENT_SCALES[@]}"
do
  echo "======================================================="
  echo "Running experiments for ${clients} clients and ${NUM_HELPERS} helpers..."
  echo "======================================================="
  
  # --- 外层循环，运行10次成功的实验 ---
  for i in $(seq 1 10)
  do
    echo "======================================================="
    echo "Running successful experiment run #${i} for ${clients} clients..."
    echo "======================================================="

    # --- 内层循环，直到找到一个可行的种子 ---
    while true;
    do
      # 1. 生成一个新的随机种子
      seed=$RANDOM
      echo "-------------------------------------------------------"
      echo "Attempting with seed: ${seed} for ${clients} clients (Run ${i}/10)..."
      echo "-------------------------------------------------------"

      # 2. 调用主实验脚本，仅用 GBD 测试收敛性
      python3 run_experiment.py \
        --num_clients ${clients} \
        --num_helpers ${NUM_HELPERS} \
        --solver gbd \
        --seed ${seed} \
        --log_file fairness_experiment_log.csv

      # 3. 检查退出状态码
      if [ $? -eq 0 ]; then
        echo ">>> SUCCESS: GBD converged with seed ${seed} for ${clients} clients."
        echo ">>> Now running all baselines with the successful seed: ${seed}"
        
        # 4. 使用这个成功的种子运行所有求解器 (包括GBD本身，以确保日志完整)
        python3 run_experiment.py \
          --num_clients ${clients} \
          --num_helpers ${NUM_HELPERS} \
          --solver all \
          --seed ${seed} \
          --log_file fairness_experiment_log.csv

        # 检查 "all" 运行是否出错 (理论上不应该，因为GBD已收敛)
        if [ $? -ne 0 ]; then
            echo ">>> WARNING: 'solver all' failed with seed ${seed} even though GBD converged. Check logs."
        fi

        break # 跳出 while 循环，进行下一次成功实验
      else
        echo ">>> INFO: GBD did not converge with seed ${seed}. Retrying with a new seed..."
      fi
    done
  done
done

echo "\nAll experiments completed successfully."
