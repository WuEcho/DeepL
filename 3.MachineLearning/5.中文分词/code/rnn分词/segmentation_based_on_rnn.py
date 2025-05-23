#coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""

class TorchModel(nn.Module):
    #input_dim:输入维度，hidden_size:隐藏层维度，num_rnn_layers:RNN层数，vocab:字表
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        #embedding层，将输入的词汇转换为向量表示，padding_idx=0表示忽略索引为0的词汇
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim, padding_idx=0) #shape=(vocab_size, dim)
        self.rnn_layer = nn.RNN(input_size=input_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_rnn_layers,
                            ) #RNN层，输入维度为input_dim，隐藏层维度为hidden_size，batch_first=True表示输入数据的第一个维度为batch_size，num_layers表示RNN的层数
        #全连接层，将RNN的输出映射到2个类别上，hidden_size表示RNN的隐藏层维度，2表示输出2个类别
        self.classify = nn.Linear(hidden_size, 2)  # w = hidden_size * 2
        #ignore_index=-100 代表忽略-100的标签，不计算loss 配合embedding的padding_idx=0
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100) 

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  #output shape:(batch_size, sen_len, hidden_size)  
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, 2) -> y_pred.view(-1, 2) (batch_size*sen_len, 2)
        if y is not None:
            #cross entropy
            #y_pred : n, class_num    [[1,2], [2,1]]
            #y      : n               [0,       1  ]

            #y:batch_size, sen_len  = 2 * 5
            #[[0,0,1,0,1],[0,1,0, -100, -100]]  y
            #[0,0,1,0,1,  0,1,0,-100.-100]    y.view(-1) shape= n = batch_size*sen_len
          
            #view(-1, 2)  -1代表自动计算，这里是batch_size*sen_len
            #y_pred.reshape(-1, 2) 将预测值展平，形状变为 (batch_size * sentence_length, 2)。
            #y.view(-1)：将真实标签展平，形状变为 (batch_size * sentence_length)。
            return self.loss_func(y_pred.reshape(-1, 2), y.view(-1))
        else:
            return y_pred

class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label) 
                self.data.append([sequence, label])
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    #将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        # 返回self.data的长度
        return len(self.data)

    def __getitem__(self, item):
        # 返回self.data中索引为item的元素
        return self.data[item]

#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence

#基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    # 创建一个Dataset对象，用于加载语料库，vocab为词汇表，max_length为最大长度
    dataset = Dataset(corpus_path, vocab, max_length) #diy __len__ __getitem__
    # 创建一个DataLoader对象，用于将Dataset对象中的数据加载到内存中，shuffle为是否打乱数据，batch_size为每个batch的大小
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #torch
    # 返回DataLoader对象
    return data_loader


def main():
    epoch_num = 5        #训练轮数
    batch_size = 20       #每次训练样本个数
    char_dim = 50         #每个字的维度
    hidden_size = 100     #隐含层维度
    num_rnn_layers = 1    #rnn层数
    max_length = 20       #样本最大长度
    learning_rate = 1e-3  #学习率
    vocab_path = "chars.txt"  #字表文件路径
    corpus_path = "../corpus.txt"  #语料文件路径
    vocab = build_vocab(vocab_path)       #建立字表
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  #建立数据集
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)     #建立优化器
    #训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()    #梯度归零
            loss = model.forward(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    return

#最终预测
def predict(model_path, vocab_path, input_strings):
    #配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 1  # rnn层数
    vocab = build_vocab(vocab_path)       #建立字表
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  #预测出的01序列
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()



if __name__ == "__main__":
    # main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    
    predict("model.pth", "chars.txt", input_strings)

