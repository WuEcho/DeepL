# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        #embedding层，将输入的词转换为向量
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        #根据配置文件中的模型类型，选择不同的编码器
        if model_type == "fast_text":
            #fast_text模型只是在embedding的基础上做，不需要中间层
            #lambda 函数代表输出就是输入，不做任何处理
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        #分类器，将编码器输出的向量转换为分类结果
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            #sequence_output:batch_size, max_len, hidden_size
            #pooler_output:batch_size, hidden_size
            x = self.encoder(x)
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)

        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        #可以采用pooling的方式得到句向量
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        #也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]

        predict = self.classify(x)   #input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        # 获取配置参数
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        # 计算padding大小
        pad = int((kernel_size - 1)/2)
        # 定义卷积层
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    # 定义一个GatedCNN类，继承自nn.Module
    def __init__(self, config):
        # 初始化函数，接收一个config参数
        super(GatedCNN, self).__init__()
        # 调用父类的初始化函数
        self.cnn = CNN(config)
        # 定义一个cnn层，使用config参数
        self.gate = CNN(config)

        # 定义一个gate层，使用config参数
    def forward(self, x):
        # 定义前向传播函数，接收一个x参数
        a = self.cnn(x)
        # 将x输入到cnn层，得到a
        b = self.gate(x)
        # 将x输入到gate层，得到b
        b = torch.sigmoid(b)
        # 对b进行sigmoid激活
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)               #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1) #二层线性
            x = self.bn_after_ff[i](x + l2)        #残差后过bn
        return x


class RCNN(nn.Module):
    # 定义RCNN类，继承自nn.Module
    def __init__(self, config):
        # 初始化函数，接收一个config参数
        super(RCNN, self).__init__()
        # 调用父类的初始化函数
        hidden_size = config["hidden_size"]
        # 从config中获取hidden_size参数
        self.rnn = nn.RNN(hidden_size, hidden_size)
        # 定义一个RNN层，输入和输出的维度都是hidden_size
        self.cnn = GatedCNN(config)
        # 定义一个GatedCNN层，传入config参数

    def forward(self, x):
        # 定义前向传播函数，接收一个x参数
        x, _ = self.rnn(x)
        # 将x传入RNN层，得到输出x和隐藏状态_
        x = self.cnn(x)
        # 将x传入GatedCNN层，得到输出x
        return x

class BertLSTM(nn.Module):
    # 定义BertLSTM类，继承自nn.Module
    def __init__(self, config):
        # 初始化函数，接收一个config参数
        super(BertLSTM, self).__init__()
        # 调用父类的初始化函数
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        # 从预训练模型中加载Bert模型，并设置return_dict为False
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)
        # 定义LSTM层，输入和输出的维度都为Bert模型的hidden_size，batch_first为True

    def forward(self, x):
        # 定义前向传播函数，接收一个x参数
        x = self.bert(x)[0]
        # 将x输入到Bert模型中，得到输出x
        x, _ = self.rnn(x)
        # 将x输入到LSTM层中，得到输出x和隐藏状态
        return x

class BertCNN(nn.Module):
    # 定义BertCNN类，继承自nn.Module
    def __init__(self, config):
        # 初始化函数，接收一个config参数
        super(BertCNN, self).__init__()
        # 调用父类的初始化函数
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        # 从预训练模型中加载Bert模型，并设置return_dict为False
        config["hidden_size"] = self.bert.config.hidden_size
        # 将Bert模型的hidden_size赋值给config
        self.cnn = CNN(config)
        # 初始化CNN模型，并传入config参数
    def forward(self, x):
        # 定义前向传播函数，接收一个x参数
        x = self.bert(x)[0]
        # 将x传入Bert模型，并获取输出
        x = self.cnn(x)
        # 将输出传入CNN模型，并获取最终输出
        return x
    
## 拿出中间层
class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        # 从预训练模型路径加载Bert模型
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        # 设置Bert模型的输出为隐藏状态
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        # 获取Bert模型的隐藏状态
        # 原本的输出是(sequence_output, pooler_output), 
        # 其中sequence_output是[batch_size, seq_len, hidden_size]的tensor，
        # 现在将每一层的结果都输出就会有三个输出，分别是embedding_output, encoder_output, hidden_states
        # hidden_states是[13, batch_size, seq_len, hidden_size]的tensor
        # 我们只需要hidden_states
        layer_states = self.bert(x)[2]#(13, batch, len, hidden)
        # 将最后两层隐藏状态相加
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    # Config["class_num"] = 3
    # Config["vocab_size"] = 20
    # Config["max_length"] = 5
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))


    # model = TorchModel(Config)
    # label = torch.LongTensor([1,2])
    # print(model(x, label))