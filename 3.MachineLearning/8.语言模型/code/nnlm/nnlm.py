#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import matplotlib.pyplot as plt

"""
基于pytorch的rnn语言模型
"""

class LanguageModel(nn.Module):
     # 初始化函数，接收输入维度和词汇表作为参数
    def __init__(self, input_dim, vocab):
         # 调用父类的初始化函数
        super(LanguageModel, self).__init__()
        # 创建一个嵌入层，将词汇表中的每个词映射到一个input_dim维度的向量
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim) 
        # 创建一个RNN层，输入维度为input_dim，输出维度为input_dim，层数为2，batch_first为True
        self.layer = nn.RNN(input_dim, input_dim, num_layers=2, batch_first=True)
        # 创建一个线性层，将input_dim维度的向量映射到len(vocab) + 1维度的向量
        self.classify = nn.Linear(input_dim, len(vocab) + 1)
        # 创建一个dropout层，dropout率为0.1
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #output shape:(batch_size, sen_len, input_dim)
        x, _ = self.layer(x)      #output shape:(batch_size, sen_len, input_dim) 
        x = x[:, -1, :]        #output shape:(batch_size, input_dim) 
        x = self.dropout(x) 
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred, y) #[1*vocab_size] []
        else:
            return torch.softmax(y_pred, dim=-1)  # 在最后一个维度上做softmax

# 说明：x = x[:, -1, :]
# x 的原始形状：(batch_size, sen_len, input_dim)
# x[:, -1, :] 表示：
​# 第一个维度：: 保留所有样本（batch_size）
​# 第二个维度：-1 取最后一个时间步的输出
​# 第三个维度：: 保留所有特征（input_dim）
​# 用途：
# 在RNN处理序列后，每个时间步都会输出一个隐藏状态
# 取最后一个时间步的输出，是因为它累积了整个序列的上下文信息
# 适用于需要基于完整序列做预测的任务（如文本分类、语言模型的下一个词预测）#


#读取语料获得字符集
#输出一份
def build_vocab_from_corpus(path):
    vocab = set()
    with open(path, encoding="utf8") as f:
        for index, char in enumerate(f.read()):
            vocab.add(char)
    vocab.add("<UNK>") #增加一个unk token用来处理未登录词
    writer = open("vocab.txt", "w", encoding="utf8")
    for char in sorted(vocab):
        writer.write(char + "\n")
    return vocab

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]        #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
        vocab["\n"] = 1
    return vocab

#加载语料
def load_corpus(path):
    return open(path, encoding="utf8").read()

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[end]
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    y = vocab[target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    # 计算困惑度
    prob = 0 
    # 将模型设置为评估模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        # 遍历句子中的每个字符
        for i in range(1, len(sentence)):
            # 计算窗口的起始位置
            start = max(0, i - window_size) #防负数
            # 获取窗口中的字符
            window = sentence[start:i]
            # 将窗口中的字符转换为索引
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            # 将索引转换为张量
            x = torch.LongTensor([x])
            # 获取目标字符的索引
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            # 如果有GPU，则将张量移动到GPU上
            if torch.cuda.is_available():
                x = x.cuda()
            # 获取模型的预测概率分布
            pred_prob_distribute = model(x)[0] #从模型的输出张量中提取当前样本的预测结果
            # 获取目标字符的概率
            target_prob = pred_prob_distribute[target_index]
            # 累加概率的对数
            prob += math.log(target_prob, 10)
    # 返回困惑度
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 10        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 128        #每个字的维度
    window_size = 6       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            watch_loss.append(loss.item())
            loss.backward()      #计算梯度
            optim.step()         #更新权重
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

#训练corpus文件夹下的所有语料，根据文件名将训练后的模型放到莫得了文件夹
def train_all():
    for path in os.listdir("corpus"):
        corpus_path = os.path.join("corpus", path)
        train(corpus_path)


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    # train("corpus.txt", True)
    train_all()