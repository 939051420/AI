import torch
from torch.utils.data import Dataset
import numpy as np


def normalization(x):
    """将数据集标准化"""
    return (x - np.mean(x)) / (np.sqrt(np.var(x)) + 1e-6)


class Cifar10Dataset(Dataset):
    # 自定义Dataset时，需要重写以下三个方法
    def __init__(self, inputs, labels, device=torch.device("cpu")):
        # 构造函数，可以自定义传入的参数，用于保存data
        super(Cifar10Dataset, self).__init__()
        self.inputs = normalization(inputs)
        self.labels = labels
        self.device = device

    def __len__(self):
        # 计算数据集数据条数
        return len(self.inputs)

    def __getitem__(self, idx):
        # 获得第idx个下标的数据
        x = torch.tensor(self.inputs[idx].reshape(3, 32, 32), dtype=torch.float32, device=self.device)  # 模型输入类型应为torch.float32
        y = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)  # torch的损失函数输入类型应为torch.long
        return x, y
