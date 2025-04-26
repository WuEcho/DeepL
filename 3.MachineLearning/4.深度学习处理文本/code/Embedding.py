#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
'''

num_embeddings = 7  #通常对于nlp任务，此参数为字符集字符总数(随机生成向量的个数)
num_embeddings_unfound = 8  #当输入字符不在字符表中时，返回的向量个数

embedding_dim = 5   #每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
print("随机初始化权重")
print(embedding_layer.weight)
print("################")

#构造字符表(构造子字符表的目的在于确定每个维度的向量对应的字符，需要自行构造或寻找第三方库支持)
vocab = { 
    "[pad]" : 0,
    "a" : 1,
    "b" : 2,
    "c" : 3,
    "d" : 4,
    "e" : 5,
    "f" : 6,
}

# 添加了对于未知字符的匹配
vocab_unfoundchar = {
    "[pad]" : 0,
    "a" : 1,
    "b" : 2,
    "c" : 3,
    "d" : 4,
    "e" : 5,
    "f" : 6,
    "[unk]" : 7,
}

def str_to_sequence(string, vocab):
    return [vocab[s] for s in string]


#padding 
def str_to_sequence_pendding(string, vocab):
    seq = [vocab[s] for s in string]
    if len(seq) < 5:
        seq += vocab["[pad]"] * (5 - len(seq))
    return seq

##cutting-out
def str_to_sequence_cut(string, vocab):
    seq = [vocab[s] for s in string]
    if len(seq) < 5:
        seq += vocab["[pad]"] * (5 - len(seq))
    elif len(seq) > 5: #这个5可以作为一个可变参数传入
        seq = seq[:5]    
    return seq

## unfound char
def str_to_sequence_unfound(string, vocab):
    seq = [vocab.get(s, vocab["[unk]"]) for s in string][:5]
    return seq

string1 = "abcde"
string2 = "ddccb"
string3 = "fedab"

# 注意如果数据中出现词表中没有的词
# 可以在词表中单独留一个位置 ”[unk]“来表示未知词 =》”[unk]“ : 7 对应的初始化向量的维度也要增加


sequence1 = str_to_sequence(string1, vocab) #[1, 2, 3, 4, 5]
sequence2 = str_to_sequence(string2, vocab) #[4, 3, 2, 4, 3]
sequence3 = str_to_sequence(string3, vocab) #[5, 4, 3, 1, 2]

print(sequence1)
print(sequence2)
print(sequence3)

x = torch.LongTensor([sequence1, sequence2, sequence3])
embedding_out = embedding_layer(x)
print(embedding_out)


# 为了让不同长度的训练样本能够放在同一个batch中(在同一批向量中在送入TensorFlow或PyTorch模型前，
# 要求张量的维度必须一致)，需要将所有样本补齐或截断到相同长度
# padding 补齐  字表中 "[pad]" : 0, 0代表补齐
# [1,2,3,0,0]
# [1,2,3,4,0]
# [1,2,3,4,5]
# 截断  在nlp任务中截断就意味着信息的丢失，如果截断的信息是有一定意义的那么就会影响模型的训练效果
# 因此在nlp任务中尽量多补零少截断（选取长度的时候选取占据数据绝不大部分的长度，一定程度上可以忽略极少部分超长数据的影响）
# [1,2,3,4,5,6,7] -> [1,2,3,4,5]
