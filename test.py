import os
import json
import torch
from torch.utils.data import DataLoader
from load_data import load_cifar10_test
from dataset import Cifar10Dataset
from tqdm import tqdm

if __name__ == "__main__":
    ckpt_path = r"output\5-21_20-18_LeNet\ckpt\LeNet_epoch10"
    device = torch.device("cuda")

    model_path = os.path.join(ckpt_path, "model.pth")
    model = torch.load(model_path).to(device)
    test_inputs, test_labels = load_cifar10_test()
    dataset = Cifar10Dataset(test_inputs, test_labels, device)
    test_dataloader = DataLoader(dataset=dataset, batch_size=100)
    correct_cnt = 0
    with torch.no_grad():  # 评估时不计算梯度
        for x, y in tqdm(test_dataloader, desc="testing"):
            out = model(x)
            num_pred = torch.argmax(out, dim=1)
            for num, label in zip(num_pred, y):
                if num == label:
                    correct_cnt += 1
    print("准确率为 ", correct_cnt / test_inputs.shape[0])
    fn = os.path.join(ckpt_path, "score.txt")
    with open(fn, "w") as f:
        dct = {
            "correct_cnt": correct_cnt,
            "total_cnt": test_inputs.shape[0],
            "score": correct_cnt / test_inputs.shape[0],
        }
        print(json.dumps(dct, indent=4), file=f)
