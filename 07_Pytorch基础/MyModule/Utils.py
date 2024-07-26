import torch
import torch.nn.functional as F


# 准确度函数（适用于2d的tensor）
# y：       神经网络原始输出（未经过softmax）
# label：   准确值
def accuracy_Fmnist(y, label):
    # 对y按行使用softmax
    y = F.softmax(y.double(), dim=1)

    # 按行找最大值下标（结果为一个一维数组）
    max_indices = torch.argmax(y, dim=1)

    # 找出所有相同元素
    same = (label == max_indices).float()

    # 计算平均值
    return torch.mean(same)


if __name__ == '__main__':
    ...