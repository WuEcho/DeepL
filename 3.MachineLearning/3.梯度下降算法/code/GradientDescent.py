import matplotlib.pyplot as pyplot
import math

# X = [0.01 * i for i in range(100)]
# Y = [2 * x ** 2 + 3 * x + 4 for x in X]
# print(X)
# print(Y)
# pyplot.scatter(X, Y)
# pyplot.show()

X = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35000000000000003, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.6900000000000001, 0.7000000000000001, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.9400000000000001, 0.9500000000000001, 0.96, 0.97, 0.98, 0.99]
Y = [4.0, 4.0302, 4.0608, 4.0918, 4.1232, 4.155, 4.1872, 4.2198, 4.2528, 4.2862, 4.32, 4.3542, 4.3888, 4.4238, 4.4592, 4.495, 4.5312, 4.5678, 4.6048, 4.6422, 4.68, 4.7181999999999995, 4.7568, 4.7958, 4.8352, 4.875, 4.9152000000000005, 4.9558, 4.9968, 5.0382, 5.08, 5.122199999999999, 5.1648, 5.2078, 5.2512, 5.295, 5.3392, 5.3838, 5.4288, 5.4742, 5.5200000000000005, 5.5662, 5.6128, 5.6598, 5.7072, 5.755, 5.8032, 5.851800000000001, 5.9008, 5.9502, 6.0, 6.0502, 6.1008, 6.1518, 6.203200000000001, 6.255000000000001, 6.3072, 6.3598, 6.4128, 6.4662, 6.52, 6.5742, 6.6288, 6.6838, 6.7392, 6.795, 6.8512, 6.9078, 6.9648, 7.022200000000001, 7.08, 7.138199999999999, 7.1968, 7.2558, 7.3152, 7.375, 7.4352, 7.4958, 7.5568, 7.6182, 7.680000000000001, 7.7422, 7.8048, 7.867800000000001, 7.9312, 7.994999999999999, 8.0592, 8.1238, 8.1888, 8.2542, 8.32, 8.3862, 8.4528, 8.5198, 8.587200000000001, 8.655000000000001, 8.7232, 8.7918, 8.8608, 8.9302]


def func(x):
    return w1 * x ** 2 + w2 * x + w3


def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

# 权重随机初始化
w1, w2, w3 = -1, 1, 0

# 学习率
lr = 0.01

for epoch in range(10000):
    epoch_loss = 0
    for x, y_true in zip(X, Y):
        y_pred = func(x)
        epoch_loss += loss(y_pred, y_true)
        #梯度计算
        grad_w1 = 2 * (y_pred - y_true) * x ** 2
        grad_w2 = 2 * (y_pred - y_true) * x
        grad_w3 = 2 * (y_pred - y_true)
        #根据梯度修改权重（优化器）
        w1 = w1 - lr * grad_w1
        w2 = w2 - lr * grad_w2
        w3 = w3 - lr * grad_w3

    epoch_loss /= len(X)
    print("第%d轮， loss %f" %(epoch, epoch_loss))
    if epoch_loss < 0.01:
        break

print("训练后权重:", w1, w2, w3)


Y1 = [func(i) for i in X]

# pyplot.scatter(X, Y, color="red")
# pyplot.scatter(X, Y1)
# pyplot.show()
