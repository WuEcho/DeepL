#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import re
import os
from nnlm import build_model, build_vocab


"""
使用训练好的语言模型
"""

def load_trained_language_model(path):
    # 加载训练好的语言模型
    char_dim = 128        #每个字的维度,与训练时保持一直
    window_size = 6       #样本文本长度,与训练时保持一直
    vocab = build_vocab("vocab.txt")      # 加载字表
    model = build_model(vocab, char_dim)  # 加载模型
    model.load_state_dict(torch.load(path))  #加载训练好的模型权重
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.window_size = window_size
    model.vocab = vocab
    return model

#计算文本ppl
def calc_perplexity(sentence, model):
    # 计算困惑度
    prob = 0
    # 不计算梯度
    with torch.no_grad():
        # 遍历句子中的每个字符
        for i in range(1, len(sentence)):
            # 计算窗口大小
            start = max(0, i - model.window_size)
            window = sentence[start:i]
            # 将窗口中的字符转换为索引
            x = [model.vocab.get(char, model.vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            # 获取目标字符的索引
            target = sentence[i]
            target_index = model.vocab.get(target, model.vocab["<UNK>"])
            # 如果有GPU，则将数据转移到GPU上
            if torch.cuda.is_available():
                x = x.cuda()
            # 获取预测的概率分布
            pred_prob_distribute = model(x)[0]
            # 获取目标字符的概率
            target_prob = pred_prob_distribute[target_index]
            # 打印窗口、目标字符和概率
            # print(window , "->", target, "prob:", float(target_prob))
            # 累加概率的对数
            prob += math.log(target_prob, 10)
    # 返回困惑度
    return 2 ** (prob * ( -1 / len(sentence)))

#加载训练好的所有模型
# 定义加载模型的函数
def load_models():
    # 获取模型文件夹中的所有文件路径
    model_paths = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/model")
    # 创建一个空字典，用于存储模型
    class_to_model = {}
    # 遍历所有文件路径
    for model_path in model_paths:
        # 获取模型名称
        class_name = model_path.replace(".pth", "")
        # 获取模型路径
        model_path = os.path.join("model", model_path)
        # 将模型名称和模型路径存入字典
        class_to_model[class_name] = load_trained_language_model(model_path)
    # 返回字典
    return class_to_model

#基于语言模型的文本分类伪代码
#class_to_model: {"class1":<language model obj1>, "class2":<language model obj2>, ..}
#每个语言模型，用对应的领域语料训练
def text_classification_based_on_language_model(class_to_model, sentence):
    ppl = []
    for class_name, class_lm in class_to_model.items():
        #用每个语言模型计算ppl
        ppl.append([class_name, calc_perplexity(sentence, class_lm)])
    ppl = sorted(ppl, key=lambda x:x[1])
    print(sentence)
    print(ppl[0: 3])
    print("==================")
    return ppl

sentence = ["在全球货币体系出现危机的情况下",
            "点击进入双色球玩法经典选号图表",
            "慢时尚服饰最大的优点是独特",
            "做处女座朋友的人真的很难",
            "网戒中心要求家长全程陪护",
            "在欧巡赛扭转了自己此前不利的状态",
            "选择独立的别墅会比公寓更适合你",
            ]


class_to_model = load_models()
for s in sentence:
    text_classification_based_on_language_model(class_to_model, s)

