#coding:utf8
import torch
import torch.nn as nn

'''
pooling层的处理
'''

#pooling操作默认对于输入张量的最后一维进行
#入参5，代表把五维池化为一维
layer = nn.AvgPool1d(4)  #AvgPool1d 一维的平均池化;也有最大池化-MaxPool1d(4)
#随机生成一个维度为3x4x5的张量
#可以想象成3条,文本长度为4,向量长度为5的样本
x = torch.rand([3, 4, 5])
print(x)
print(x.shape)
x = x.transpose(1,2) #转置的是文本长度和向量长度 3表示的是bitch_size 属于第0维
print(x.shape, "交换后") #[3,5,4]
#经过pooling层
y = layer(x)
print(y)    
print(y.shape) #[3,5,1] 
#squeeze方法去掉值为1的维度
y = y.squeeze()
print(y)
print(y.shape)
