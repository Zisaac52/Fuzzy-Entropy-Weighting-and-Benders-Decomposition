import requests
import torch

from main_node import getNetGlob
from models.Nets import CNN
from utils.dataset import test_img
from utils.models import hexToStateDict
from utils.options import args_debug_parser
from torchvision import datasets, transforms

args = args_debug_parser()

# 获取训练信息，从本地服务器获取
def getTrainInfo():
    resp = requests.get(f'http://localhost:{args.port}/getTrainInfo')
    data = resp.json()
    return data['result']

# 获取全局空模型
def getNetGlobEmpty():
    trainInfo = getTrainInfo()
    if trainInfo['model_name'] == "cnn":
        model = CNN(num_channels=trainInfo['num_channels'], num_classes=trainInfo['num_classes'])
    return model

# 通过块号获取模型
def getModelByBlockNumber(blockNumber):
    model = getNetGlobEmpty()
    resp = requests.get(f'http://localhost:{args.port}/getGlobalModelByBlock/{blockNumber}')
    result = resp.json()
    try:
        stateDict = hexToStateDict(result['global_model_hex'])
        model.load_state_dict(stateDict)
    except Exception as e:
        print(f'Error loading model state dict: {e}')
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == "cifar":
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit("Dataset not supported")

    for i in range(103):
        net_glob = getModelByBlockNumber(i).to(device=device)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args.epoch, args.gpu)
        print(f'Block: {i}, Accuracy: {acc_test}, Loss: {loss_test}')
