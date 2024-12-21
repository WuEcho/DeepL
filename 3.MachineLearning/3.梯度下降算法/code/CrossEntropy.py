import torch
import torch.nn as nn
import numpy as np

'''
手动实现交叉熵的计算
'''

#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]])
#正确的类别分别为1,2,0
target = torch.LongTensor([1, 2, 0])
#这里是longTensor是因为在 nn.CrossEntropyLoss 中，真实类别标签（target）需要满足以下条件：
#类型：必须是整型张量（torch.int64 或 torch.LongTensor）。
#值的范围：必须是 0 到 C-1（类别总数减一）。
loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")


#实现softmax函数
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

#验证softmax函数
# print(torch.softmax(pred, dim=1))
# print(softmax(pred.numpy()))


#将输入转化为onehot矩阵
#to_one_hot 是一种将类别标签转换为独热编码（one-hot encoding）的操作。
#类别标签：通常是一个索引，例如 [1, 2, 0]，其中每个值表示类别索引。
#独热编码：将类别标签转为向量形式，例如：
#类别 1 转为 [0, 1, 0]
#类别 2 转为 [0, 0, 1]
#类别 0 转为 [1, 0, 0]
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

#手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size

print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")
