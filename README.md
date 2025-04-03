# BAFL-Entropy: Blockchain-Based Asynchronous Federated Learning with Entropy Weighting

本项目实现了基于区块链的异步联邦学习框架（BAFL），并结合了信息熵加权方法（IEWM）以及传统的 FedAsync 聚合策略。代码主要参考了以下论文：

```
@article{REPO-283,
    author = "Feng, Lei and Zhao, Yiqi and Guo, Shaoyong and Qiu, Xuesong and Li, Wenjing and Yu, Peng",
    journal = "IEEE Transactions on Computers",
    number = "5",
    pages = "1092--1103",
    publisher = "IEEE",
    title = "{BAFL: A Blockchain-Based Asynchronous Federated Learning Framework}",
    volume = "71",
    year = "2021"
}
```

## 环境设置

1.  **克隆仓库** (如果尚未完成):
    ```bash
    git clone https://gitee.com/zclisaac/afl-fuzzy-entropy.git
    cd BAFL-Entropy
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

项目提供了脚本来运行单个或批量的联邦学习实验。

### 运行单个实验

使用 `run_experiment.sh` 脚本来运行具有特定配置的单个实验。

**用法:**

```bash
./run_experiment.sh [--node <num_nodes>] [--epoch <num_epochs>] [--method <aggregate_method>]
```

**参数:**

*   `--node <num_nodes>`: 参与联邦学习的节点数量 (默认为 4)。
*   `--epoch <num_epochs>`: 每个节点在每一轮聚合前本地训练的轮数 (默认为 1)。
*   `--method <aggregate_method>`: 使用的聚合方法。可选值:
    *   `fuzzy` (默认): 模糊熵熵权法。
    *   `iewm`: 信息熵加权方法。
    *   `fedasync`: 简单的异步联邦平均。
*   `--fuzzy_m <value>`: 当 `--method` 为 `fuzzy` 时，指定模糊熵计算中的 `m` 参数 (默认为 2，如果未通过命令行指定)。

**示例:**

*   运行一个使用 8 个节点、每个节点训练 5 轮、采用 Fuzzy 方法、`fuzzy_m` 设为 3 的实验：
    ```bash
    ./run_experiment.sh --node 8 --epoch 5 --method fuzzy --fuzzy_m 3
    ```
*   运行一个使用默认配置（4 个节点，1 轮，Fuzzy 方法，默认 `fuzzy_m`）的实验：
    ```bash
    ./run_experiment.sh
    ```

**输出:**

*   实验日志将保存在 `logs_method_<method>_node_<nodes>_epoch_<epochs>/` 目录下，包括服务器日志 (`server.log`) 和每个节点的日志 (`node_*.log`)。
*   最终模型的测试准确率将记录在 `experiment_results.csv` 文件中。

### 运行批量实验 (针对 8 个节点)

使用 `run_all_experiments_node8.sh` 脚本可以方便地为 8 个节点运行一系列实验，遍历 1 到 10 个训练轮数、所有支持的聚合方法 (`fuzzy`, `iewm`, `fedasync`)，以及在 `fuzzy` 方法下预定义的 `fuzzy_m` 值 (在脚本内部的 `FUZZY_M_VALUES` 数组中指定)。

**用法:**

```bash
./run_all_experiments_node8.sh
```

该脚本会自动调用 `run_experiment.sh` 来执行每个配置组合。所有结果同样会记录在 `experiment_results.csv` 中。

## 支持的配置

*   **数据集**:
    *   MNIST (默认)
    *   CIFAR-10 (目前仅支持 IID 数据划分)
    *   *注意*: 数据集在 `main_node.py` 中配置，目前脚本默认使用 MNIST。
*   **聚合方法**:
    *   `fuzzy`
    *   `iewm`
    *   `fedasync`
*   **数据划分**:
    *   IID (目前脚本默认使用 IID 划分)
    *   Non-IID (代码中存在支持，但当前脚本未配置为使用 Non-IID)

## 文件结构概览

*   `main_server.py`: 联邦学习中心服务器逻辑。
*   `main_node.py`: 联邦学习客户端（节点）逻辑。
*   `node.py`: 节点训练的核心类。
*   `models/`: 包含神经网络模型 (`Nets.py`) 和联邦学习聚合逻辑 (`Fed.py`)。
*   `utils/`: 包含数据处理 (`dataset.py`, `sampling.py`)、模型序列化 (`models.py`) 和命令行参数解析 (`options.py`) 等工具函数。
*   `run_experiment.sh`: 运行单个实验的脚本。
*   `run_all_experiments_node8.sh`: 运行批量实验的脚本 (针对 8 节点)。
*   `view_aggregated_model.py`: 用于在实验结束后评估最终全局模型的脚本。
*   `requirements.txt`: 项目依赖库列表。
*   `experiment_results.csv`: 存储实验结果的 CSV 文件。
*   `logs_*/`: 存储实验日志的目录。
