import numpy as np
import pickle
import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")

def load_cifar10_file(file_name):
    """读取cifar-10数据集的二进制文件"""
    with open(file_name, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    inputs = data[b'data'].astype(int)
    labels = np.array(data[b'labels'])
    return inputs, labels


def load_cifar10_train():
    """
    读取cifar-10训练集
    returns:
    train_inputs: ndArray(shape=(50000, 3072))
    train_labels: ndArray(shape=(50000,))
    """
    inputs_list, labels_list = [], []
    for i in range(1, 6):
        batch_fn = os.path.join("cifar-10-batches-py", f"data_batch_{i}")
        inputs, labels = load_cifar10_file(batch_fn)
        inputs_list.append(inputs)
        labels_list.append(labels)
    train_inputs = np.concatenate(inputs_list, axis=0)
    train_labels = np.concatenate(labels_list, axis=0)
    return train_inputs, train_labels


def load_cifar10_test():
    """
    读取Cifar-10测试集
    returns:
    test_inputs: ndArray(shape=(10000, 3072))
    test_labels: ndArray(shape=(10000,))
    """
    test_file_name = os.path.join("cifar-10-batches-py", "test_batch")
    test_inputs, test_labels = load_cifar10_file(test_file_name)
    return test_inputs, test_labels


def draw_img(data, n_row, n_col):
    """可视化训练集，传入train_inputs，绘制(n_row * n_col)个数据"""
    fig, axes = plt.subplots(n_row, n_col, figsize=(16, 9))  # 创建一个2x3的图像布局
    for i in range(n_row):
        for j in range(n_col):
            # 由于数据集中的维度是(rgb, 列, 行), 而plt中是是(列, 行, rgb), 因此需要将3 * 32 * 32转化为32 * 32 * 3
            # plt.imshow中rgb图像是int类型[0,255]或者float类型[0,1]
            img = np.transpose(data[i * n_col + j].reshape(3, 32, 32), (1, 2, 0)).astype(int)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_axis_off()  # 关闭坐标轴
    os.makedirs("output", exist_ok=True)
    fn = os.path.join("output", "dataset.png")
    plt.savefig(fn)  # plt.savefig必须在plt.show之前，否则会清空画布
    plt.show()


if __name__ == "__main__":
    #  运行该脚本，可视化数据集
    train_inputs, train_labels = load_cifar10_train()
    draw_img(train_inputs, 9, 16)
