import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import Cifar10Dataset
from draw_loss import draw_loss
from load_data import load_cifar10_train
from model import LeNet


def set_seed(seed):
    """设置随机数种子, 保证结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_config(args):
    """初始化训练配置"""
    set_seed(args.seed)
    config = CONFIG
    # 保存目录包含时间信息，避免版本混乱
    t = datetime.datetime.now()
    save_dir = os.path.join(args.output_dir, f"{t.month}-{t.day}_{t.hour:02d}-{t.minute:02d}_{args.save_name}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存本次训练的参数
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        print(json.dumps(config, indent=4), file=f)
    return config, save_dir


def initialize_model(args, model_config):
    """初始化模型和损失函数"""
    # 初始化模型
    if not args.model_path:  # 随机初始化模型
        print("Creating model")
        model = LeNet(model_config)
    else:  # 加载模型参数
        print(f"Loading model from: {args.model_path}")
        model = torch.load(args.model_path)
    model = model.to(torch.device(args.device))
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    return model, loss_function


def prepare_dataloader(args, training_config):
    """准备数据加载器"""
    print("Loading dataset")
    train_inputs, train_labels = load_cifar10_train()
    dataset = Cifar10Dataset(train_inputs, train_labels, torch.device(args.device))
    train_dataloader = DataLoader(dataset=dataset, batch_size=training_config["batch_size"])
    return train_dataloader


def save_checkpoint(model, args, epoch, losses, save_dir):
    """保存模型和训练损失"""
    ckpt_dir = os.path.join(save_dir, args.ckpt_dir, f"{args.save_name}_epoch{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保存模型参数
    torch.save(model, os.path.join(ckpt_dir, "model.pth"))

    # 保存损失函数
    loss_file_name = os.path.join(ckpt_dir, "loss_list.json")
    with open(loss_file_name, "w") as f:
        print(json.dumps(losses), file=f)

    draw_loss(ckpt_dir)


def train_model(model, loss_function, train_dataloader, training_config, args, save_dir):
    """模型训练循环"""
    losses = []  # 每个step中的损失函数
    pbar = tqdm(total=args.max_epochs * len(train_dataloader), ncols=95)  # 进度条
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    for epoch in range(1, args.max_epochs + 1):
        for batch_id, (inputs, label) in enumerate(train_dataloader):
            inputs = inputs
            outputs = model(inputs)  # 模型前向传播，计算输出
            loss = loss_function(outputs, label)  # 计算损失函数，outputs自动进行softmax，label自动转换为独热码
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新优化器参数
            losses.append(loss.item())  # 记录当前 step 的损失函数

            # 更新进度条
            pbar.update()
            pbar.set_description(
                f"epoch:{epoch},batch:{batch_id + 1}/{len(train_dataloader)},loss:{np.mean(losses[-50:]):.6f}")

        # 每训练一定数量的epoch保存一次模型参数
        if epoch % args.save_epochs == 0:
            save_checkpoint(model, args, epoch, losses, save_dir)

    if args.max_epochs % args.save_epochs != 0:
        save_checkpoint(model, args, args.max_epochs, losses, save_dir)

    pbar.close()  # 关闭进度条，避免后续print时打印混乱


def get_args():
    """获得运行参数"""
    parser = argparse.ArgumentParser()
    # model params
    parser.add_argument("--model_path", type=str, default=None, help="the path to load model")
    # train params
    parser.add_argument("--max_epochs", type=int, default=200, help="max epochs to run dataloader")
    parser.add_argument("--seed", type=int, default=123456, help="the random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    # save params
    parser.add_argument("--save_epochs", type=int, default=10, help="how many epochs to save a model")
    parser.add_argument("--output_dir", type=str, default="output", help="the dir to save outputs")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="the subdir to save checkpoints")
    parser.add_argument("--save_name", type=str, default="LeNet", help="save model in: ./output_dir/time+save_name/ckpt_dir/save_name")
    args = parser.parse_args()
    print("args =", args)
    return args


def main():
    args = get_args()  # 获得参数
    config, save_dir = initialize_config(args)  # 加载配置
    model, loss_function = initialize_model(args, config["model_config"])  # 加载模型
    train_dataloader = prepare_dataloader(args, config["training_config"])  # 加载数据集
    train_model(model, loss_function, train_dataloader, config["training_config"], args, save_dir)  # 训练模型


if __name__ == "__main__":
    main()
