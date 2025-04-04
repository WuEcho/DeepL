import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
'''

关于transformers自带的序列化工具
模型文件下载 https://huggingface.co/models

'''
##使用bert必须要搭配bert预训练使用的词表以及分词方式

##加载预训练模型
# bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)

#分词器
tokenizer = BertTokenizer.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese")

string = "咱呀么老百姓今儿个真高兴"
#分字
tokens = tokenizer.tokenize(string)
print("分字：", tokens)
#编码，前后自动添加了[cls]和[sep]
encoding = tokenizer.encode(string)
print("编码：", encoding)
#文本对编码, 形式[cls] string1 [sep] string2 [sep]
string1 = "今天天气真不错"
string2 = "明天天气怎么样"
encoding = tokenizer.encode(string1, string2)
print("文本对编码：", encoding)
#同时输出attention_mask和token_type编码
encoding = tokenizer.encode_plus(string1, string2)
print("全部编码：", encoding)


