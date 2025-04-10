# AFL-Fuzzy-Entropy: Asynchronous Federated Learning with Fuzzy Entropy Weighting

本项目旨在研究和实现一种基于模糊熵加权的异步联邦学习（Asynchronous Federated Learning, AFL）聚合策略。通过为每个本地模型分配基于其模糊熵计算得出的权重，期望能够提高模型聚合的效率和最终模型的性能，特别是在非独立同分布（Non-IID）数据场景下。

## 环境设置

1.  **克隆仓库**:
    ```bash
    git clone https://gitee.com/zclisaac/afl-fuzzy-entropy.git
    cd afl-fuzzy-entropy
    ```

2.  **安装依赖**:
    建议使用 Python 虚拟环境 (如 `venv` 或 `conda`)。
    ```bash
    # 创建并激活虚拟环境 (可选但推荐)
    # python3 -m venv venv
    # source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows

    # 安装所需的库
    pip install -r requirements.txt
    ```
    *注意*: `requirements.txt` 文件包含了运行此项目所需的所有 Python 库。

## 运行实验

项目提供了脚本来运行单个或批量的联邦学习实验，支持 CPU 和 GPU。

### 运行单个实验 (CPU)

使用 `run_experiment.sh` 脚本来运行具有特定配置的单个实验（使用 CPU）。

**用法:**

```bash
./run_experiment.sh [--node <num_nodes>] [--epoch <num_epochs>] [--method <aggregate_method>] [--fuzzy_m <m_value>] [--fuzzy_r <r_value>]
```

**参数:**

*   `--node <num_nodes>`: 参与联邦学习的节点数量 (默认为 4)。
*   `--epoch <num_epochs>`: 每个节点在每一轮聚合前本地训练的轮数 (默认为 1)。
*   `--method <aggregate_method>`: 使用的聚合方法。可选值:
    *   `fuzzy` (默认): 模糊熵熵权法。
    *   `iewm`: 信息熵加权方法。
    *   `fedasync`: 简单的异步联邦平均。
*   `--fuzzy_m <m_value>`: 当 `--method` 为 `fuzzy` 时，指定模糊熵计算中的 `m` 参数 (默认为 2)。
*   `--fuzzy_r <r_value>`: 当 `--method` 为 `fuzzy` 时，指定模糊熵计算中的 `r` 参数 (默认为 0.5)。

**示例:**

*   运行一个使用 8 个节点、每个节点训练 5 轮、采用 Fuzzy 方法、`m=3`, `r=0.7` 的实验：
    ```bash
    ./run_experiment.sh --node 8 --epoch 5 --method fuzzy --fuzzy_m 3 --fuzzy_r 0.7
    ```
*   运行一个使用默认配置（4 个节点，1 轮，Fuzzy 方法，默认 `m`, `r`）的实验：
    ```bash
    ./run_experiment.sh
    ```

**输出:**

*   实验日志将保存在 `logs/logs_method_<method>_node_<nodes>_epoch_<epochs>.../` 目录下。
*   最终模型的测试准确率将记录在 `experiment_results.csv` 文件中。

### 运行单个实验 (GPU)

使用 `run_gpu_experiment.sh` 脚本来运行具有特定配置的单个实验（使用 GPU）。

**用法:**

```bash
./run_gpu_experiment.sh [--gpu <gpu_id>] [--node <num_nodes>] [--epoch <num_epochs>] [--method <aggregate_method>] [--fuzzy_m <m_value>] [--fuzzy_r <r_value>]
```

**参数:**

*   `--gpu <gpu_id>`: 指定使用的 GPU ID (默认为 0)。
*   其他参数 (`--node`, `--epoch`, `--method`, `--fuzzy_m`, `--fuzzy_r`) 与 CPU 版本脚本相同。

**示例:**

*   在 GPU 1 上运行一个使用 4 个节点、每个节点训练 3 轮、采用 Fuzzy 方法、`m=4`, `r=0.7` 的实验：
    ```bash
    ./run_gpu_experiment.sh --gpu 1 --node 4 --epoch 3 --method fuzzy --fuzzy_m 4 --fuzzy_r 0.7
    ```

**输出:**

*   实验日志将保存在 `logs/logs_gpu<gpu_id>_method_<method>_node_<nodes>_epoch_<epochs>.../` 目录下。
*   最终模型的测试准确率将记录在 `gpu_experiment_results.csv` 文件中。

### 运行批量实验 (针对 8 个节点)

*   **CPU**: 使用 `run_all_experiments_node8.sh` 脚本可以方便地为 8 个节点运行一系列 CPU 实验，遍历 1 到 10 个训练轮数、所有支持的聚合方法 (`fuzzy`, `iewm`, `fedasync`)，以及预定义的 `fuzzy_m` 和 `fuzzy_r` 值 (在脚本内部指定)。
    ```bash
    ./run_all_experiments_node8.sh
    ```
*   **GPU**: 使用 `run_all_gpu_experiments_node8.sh` 脚本可以方便地为 8 个节点运行一系列 GPU 实验。该脚本会为每个实验配置自动分配 GPU ID (从 0 开始轮换)。
    ```bash
    ./run_all_gpu_experiments_node8.sh
    ```

所有批量实验的结果将分别记录在 `experiment_results.csv` (CPU) 和 `gpu_experiment_results.csv` (GPU) 中。

### 参数网格搜索 (Fuzzy Entropy - GPU)

为了系统地评估模糊熵参数 `m` 和 `r` 对模型性能的影响，项目提供了网格搜索脚本。

**目的**: 寻找在特定实验设置（如节点数、本地训练轮数）下最优的 `m` 和 `r` 组合。

**运行网格搜索脚本**:

*   使用 `run_grid_search_gpu.sh` 脚本在 GPU 上执行网格搜索。
    ```bash
    ./run_grid_search_gpu.sh
    ```
*   脚本内部定义了固定的实验参数（例如，当前设置为 GPU 0, 8 个节点, 2 个本地轮次）以及需要遍历的 `m` 和 `r` 值列表。
*   每次实验运行后，其结果（准确率、损失率等）会被提取并追加到特定的 CSV 文件中，文件名格式为 `gpu_grid_search_results_node<节点数>_epoch<轮数>.csv` (例如 `gpu_grid_search_results_node8_epoch2.csv`)。

**分析网格搜索结果**:

*   使用 `analyze_grid_search.py` 脚本来分析网格搜索生成的 CSV 结果文件。
    ```bash
    python analyze_grid_search.py [--file <results_csv_file>] [--nodes <num_nodes>] [--epochs <num_epochs>]
    ```
*   脚本默认会读取与当前设置（8 节点, 2 轮）对应的结果文件 (`gpu_grid_search_results_node8_epoch2.csv`)。您可以通过 `--file`, `--nodes`, `--epochs` 参数指定不同的文件或设置。
*   脚本会生成准确率和损失率的热力图、轮廓图和三维表面图，以可视化不同 (m, r) 组合下的性能，并将图表保存在 `plots_node<节点数>_epoch<轮数>/` 目录下 (例如 `plots_node8_epoch2/`)。

**结果摘要 (示例: 8 节点, 2 轮)**:

*   在 8 节点、2 本地轮次的设置下，观察到表现较优的参数组合为 (m=5, r=0.7) 和 (m=3, r=0.2)。减少本地训练轮数有助于放大不同参数组合间的性能差异。

## 支持的配置

*   **数据集**:
    *   MNIST (默认)
    *   CIFAR-10 (目前仅支持 IID 数据划分)
    *   *注意*: 数据集在 `utils/options.py` 中通过 `--dataset` 参数配置，目前脚本默认使用 MNIST。
*   **聚合方法**:
    *   `fuzzy` (模糊熵加权)
    *   `iewm` (信息熵加权)
    *   `fedasync` (简单异步平均)
*   **数据划分**:
    *   IID (目前脚本默认使用 IID 划分)
    *   Non-IID (代码中存在支持，可通过 `--iid` 参数设为 `False` 来启用，但当前运行脚本未显式配置 Non-IID 运行)

## 文件结构概览

*   `main_server.py`: 联邦学习中心服务器逻辑。
*   `main_node.py`: 联邦学习客户端（节点）逻辑。
*   `node.py`: 节点训练的核心类。
*   `models/`:
    *   `Nets.py`: 包含神经网络模型定义。
    *   `Fed.py`: 包含联邦学习聚合逻辑 (FedAvg, IEWM, Fuzzy)。
*   `utils/`:
    *   `options.py`: 命令行参数解析。
    *   `dataset.py`: 数据集加载和预处理。
    *   `sampling.py`: 数据划分 (IID, Non-IID)。
    *   `models.py`: 模型序列化/反序列化工具。
*   `run_experiment.sh`: 运行单个 CPU 实验的脚本。
*   `run_gpu_experiment.sh`: 运行单个 GPU 实验的脚本。
*   `run_all_experiments_node8.sh`: 运行批量 CPU 实验的脚本 (针对 8 节点)。
*   `run_all_gpu_experiments_node8.sh`: 运行批量 GPU 实验的脚本 (针对 8 节点)。
*   `run_grid_search_gpu.sh`: 运行模糊熵参数网格搜索的脚本 (GPU)。
*   `analyze_grid_search.py`: 分析网格搜索结果并生成图表的脚本。
*   `view_aggregated_model.py`: 用于在实验结束后评估最终全局模型的脚本。
*   `requirements.txt`: 项目依赖库列表。
*   `experiment_results.csv`: 存储 CPU 实验结果的 CSV 文件。
*   `gpu_experiment_results.csv`: 存储 GPU 实验结果的 CSV 文件。
*   `logs/`: 存储实验日志的目录。
*   `.gitignore`: 指定 Git 忽略的文件和目录。
*   `README.md`: 本文件。
