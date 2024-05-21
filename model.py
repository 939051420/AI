import os

import torch.nn as nn
from config import CONFIG


class LeNet(nn.Module):
    def __init__(self, config):  # 构造函数，用于定义网络结构，自定义时可以使用config中的参数
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if config["hidden_act"] == "relu":
            self.act = nn.functional.relu
        elif config["hidden_act"] == "silu":
            self.act = nn.functional.silu
        elif config["hidden_act"] == "sigmoid":
            self.act = nn.functional.sigmoid
        elif config["hidden_act"] == "tanh":
            self.act = nn.functional.tanh
        else:
            raise NotImplementedError

    def forward(self, x):  # 定义前向传播函数
        # x (batch, 3, 32, 32)
        x = self.act(self.conv1(x))  # output(batch, 16, 28, 28)
        x = self.pool1(x)  # output(batch, 16, 14, 14)
        x = self.act(self.conv2(x))  # output(batch, 32, 10, 10)
        x = self.pool2(x)  # output(batch, 32, 5, 5)
        x = x.view(x.size(0), -1)  # output(batch, 32 * 5 * 5)
        x = self.act(self.fc1(x))  # output(batch, 120)
        x = self.act(self.fc2(x))  # output(batch, 84)
        x = self.fc3(x)  # output(batch, 10)
        return x


if __name__ == "__main__":
    # 运行该脚本，打印模型信息
    model = LeNet(CONFIG["model_config"])
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "model.txt"), "w") as f:
        print(model, file=f)
        print("------------------------------------------", file=f)
        total_params = sum(p.numel() for p in model.parameters())
        print("total parameters =", total_params, file=f)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable parameters =", trainable_params, file=f)
        print("------------------------------------------", file=f)
        print("state_dict:", file=f)
        state_dict = model.state_dict()
        for k, v in state_dict.items():
            print(f"{k:15s} {v.shape}", file=f)
