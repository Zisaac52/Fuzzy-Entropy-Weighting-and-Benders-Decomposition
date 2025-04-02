import argparse # <-- 添加导入
import requests
import torch
from models.Nets import CNN
from utils.models import hexToStateDict
from torchvision import datasets, transforms
from utils.dataset import test_img
from torch.utils.data import DataLoader

# 添加参数解析器
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--port', type=int, default=8000, help="Server port")
    args = parser.parse_args()
    return args

# 从服务器获取聚合后的模型
def getAggregatedModel(port): # <-- 添加 port 参数
    # 使用传入的 port 构建 URL
    resp = requests.get(f'http://localhost:{port}/newGlobalModel')

    # 检查响应状态码
    if resp.status_code != 200:
        print(f"Failed to retrieve model. Status code: {resp.status_code}")
        print(f"Response text: {resp.text}")
        return None
    
    try:
        response_json = resp.json()
        print(f"Response JSON: {response_json}")  # 添加调试信息，查看服务器返回的内容
        aggregated_model_hex = response_json['global_model_hex']
        return aggregated_model_hex
    except KeyError:
        print("Error: 'global_model_hex' not found in the response.")
        print(f"Response content: {response_json}")
        return None
    except ValueError as e:
        print("Error decoding JSON response.")
        print(f"Response text: {resp.text}")
        return None



# 加载模型
def loadModelFromServer(port): # <-- 添加 port 参数
    # 将 port 传递给 getAggregatedModel
    aggregated_model_hex = getAggregatedModel(port)

    if aggregated_model_hex is None:
        print("Failed to load the model from server.")
        return None

    try:
        # 假设服务器返回的模型为 CNN
        model = CNN(num_channels=1, num_classes=10)  # 根据您的模型架构调整
        state_dict = hexToStateDict(aggregated_model_hex)  # 解码为 PyTorch 模型参数字典
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == '__main__':
    args = args_parser() # <-- 解析参数
    # 设备选择 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device: {device}") # <-- 打印使用的设备

    # 修改 loadModelFromServer 调用以传递端口
    model = loadModelFromServer(args.port)

    if model is None:
        print("Model loading failed.")
    else:
        model.to(device) # 将模型移动到选择的设备
        model.eval()

        # 加载测试数据集
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        # 使用测试函数进行模型评估
        batch_size = 64
        # 将 args.gpu 传递给 test_img
        acc_test, loss_test = test_img(model, dataset_test, batch_size, gpu=args.gpu)

        print(f"Global Model Test: Accuracy: {acc_test * 100:.2f}%, Average Loss: {loss_test:.4f}")

    # 移除了重复的 device 打印
