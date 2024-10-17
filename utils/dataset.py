from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_img(net_g, datatest, bs, gpu):
    net_g.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0

    # 根据传入的 batch_size 创建数据加载器
    data_loader = DataLoader(datatest, batch_size=bs)
    
    # 遍历测试数据集
    for idx, (data, target) in enumerate(data_loader):
        if gpu != -1:
            data, target = data.to(torch.device("cuda")), target.to(torch.device("cuda"))  # 使用GPU
        else:
            data, target = data.to(torch.device("cpu")), target.to(torch.device("cpu"))  # 使用CPU
        
        log_probs = net_g(data)  # 模型前向传播
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()  # 计算损失
        
        # 获取预测值
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()  # 计算预测正确的样本数

    # 计算平均损失和准确率
    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)  # 保持准确率为比例值

    # 输出测试集结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy * 100:.2f}%)\n')

    return accuracy, test_loss
