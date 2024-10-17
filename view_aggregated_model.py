import requests
import torch
from models.Nets import CNN
from utils.models import hexToStateDict
from torchvision import datasets, transforms
from utils.dataset import test_img
from torch.utils.data import DataLoader

# 从服务器获取聚合后的模型
def getAggregatedModel():
    resp = requests.get('http://localhost:8000/newGlobalModel')
    
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
def loadModelFromServer():
    aggregated_model_hex = getAggregatedModel()

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
    device = torch.device('cpu')  # 明确指定使用CPU
    model = loadModelFromServer()

    if model is None:
        print("Model loading failed.")
    else:
        model.to(device)  # 将模型移动到CPU
        model.eval()

        # 加载测试数据集
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        # 使用测试函数进行模型评估
        batch_size = 64
        acc_test, loss_test = test_img(model, dataset_test, batch_size, gpu=-1)  # 确保gpu参数为-1

        print(f"Global Model Test: Accuracy: {acc_test * 100:.2f}%, Average Loss: {loss_test:.4f}")

    print(f"Using device: {device}")
