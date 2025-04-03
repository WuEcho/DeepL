# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        # 初始化函数，传入数据路径和配置参数
        self.config = config
        # 将配置参数赋值给实例变量
        self.path = data_path
        # 将数据路径赋值给实例变量
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        # 定义标签到索引的映射关系
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        # 定义索引到标签的映射关系
        self.config["class_num"] = len(self.index_to_label)
        # 将标签数量赋值给配置参数
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        # 如果模型类型为bert，则加载预训练模型
        self.vocab = load_vocab(config["vocab_path"])
        # 加载词汇表
        self.config["vocab_size"] = len(self.vocab)
        # 将词汇表大小赋值给配置参数
        self.load()


    def load(self):
        # 初始化data为空列表
        self.data = []
        # 打开文件，以utf8编码读取
        with open(self.path, encoding="utf8") as f:
            # 遍历文件中的每一行
            for line in f:
                # 将每一行转换为json格式
                line = json.loads(line)
                # 获取tag
                tag = line["tag"]
                # 将tag转换为对应的索引
                label = self.label_to_index[tag]
                # 获取title
                title = line["title"]
                # 判断模型类型
                if self.config["model_type"] == "bert":
                    # 如果是bert模型，将title转换为对应的input_id
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    # 否则，将title转换为对应的input_id
                    input_id = self.encode_sentence(title)
                # 将input_id转换为LongTensor类型
                input_id = torch.LongTensor(input_id)
                # 将label转换为LongTensor类型
                label_index = torch.LongTensor([label])
                # 将input_id和label_index添加到data中
                self.data.append([input_id, label_index])
        # 返回data
        return

    def encode_sentence(self, text):
        # 将输入的文本转换为对应的id
        input_id = []
        for char in text:
            # 获取每个字符对应的id，如果字符不在词汇表中，则使用[UNK]对应的id
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # 对id进行填充，使其长度与最大长度相同
        input_id = self.padding(input_id)
        # 返回填充后的id
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    # 定义一个方法，用于返回对象的长度
    def __len__(self):
        # 返回对象的data属性的长度
        return len(self.data)

    def __getitem__(self, index):
        # 根据索引返回数据
        return self.data[index]

def load_vocab(vocab_path):
    # 定义一个空字典，用于存储词汇表
    token_dict = {}
    # 打开词汇表文件
    with open(vocab_path, encoding="utf8") as f:
        # 遍历文件中的每一行
        for index, line in enumerate(f):
            # 去除行末的换行符
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
# 定义一个函数，用于加载数据
def load_data(data_path, config, shuffle=True):
    # 创建一个DataGenerator对象，用于生成数据
    dg = DataGenerator(data_path, config)
    # 创建一个DataLoader对象，用于加载数据，batch_size为config中的batch_size，shuffle为True或False
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # 返回DataLoader对象
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
