#coding:utf8
import jieba
import math
import os
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk

"""
基于tfidf实现文本相似度计算
"""

jieba.initialize()

#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
#之后统计每篇文档重要在前10的词，统计出重要词词表
#重要词词表用于后续文本向量化
# 定义一个函数，用于加载数据
def load_data(file_path):
    # 创建一个空列表，用于存储文档
    corpus = []
    # 打开文件，以utf8编码读取
    with open(file_path, encoding="utf8") as f:
        # 将文件内容转换为json格式
        documents = json.loads(f.read())
        # 遍历文档
        for document in documents:
            # 将文档的标题和内容拼接起来，并添加到corpus列表中
            corpus.append(document["title"] + "\n" + document["content"])
    # 调用calculate_tfidf函数，计算tf-idf值
    tf_idf_dict = calculate_tfidf(corpus)
    # 调用tf_idf_topk函数，获取前5个关键词
    topk_words = tf_idf_topk(tf_idf_dict, top=5, print_word=False)
    # 创建一个空集合，用于存储关键词
    vocab = set()
    # 遍历topk_words
    for words in topk_words.values():
        # 遍历关键词和对应的分数
        for word, score in words:
            # 将关键词添加到vocab集合中
            vocab.add(word)
    # 打印词表大小
    print("词表大小：", len(vocab))
    # 返回tf-idf字典、词表和corpus列表
    return tf_idf_dict, list(vocab), corpus


#passage是文本字符串
#vocab是词列表
#向量化的方式：计算每个重要词在文档中的出现频率
# 定义一个函数，将文档转换为向量
def doc_to_vec(passage, vocab):
    # 初始化一个向量，长度为词汇表长度，初始值为0
    vector = [0] * len(vocab)
    # 使用jieba分词，将文档分词
    passage_words = jieba.lcut(passage)
    # 遍历词汇表
    for index, word in enumerate(vocab):
        # 计算每个词在文档中出现的次数，并除以文档的总词数，得到词的权重
        vector[index] = passage_words.count(word) / len(passage_words)
    # 返回向量
    return vector

#先计算所有文档的向量
# 定义一个函数，用于计算语料库中的向量
def calculate_corpus_vectors(corpus, vocab):
    # 将语料库中的每个文档转换为向量
    corpus_vectors = [doc_to_vec(c, vocab) for c in corpus]
    # 返回语料库中的向量
    return corpus_vectors

#计算向量余弦相似度
# 定义一个计算两个向量余弦相似度的函数
def cosine_similarity(vector1, vector2):
    # 计算两个向量的点积
    x_dot_y = sum([x*y for x, y in zip(vector1, vector2)])
    # 计算第一个向量的模
    sqrt_x = math.sqrt(sum([x ** 2 for x in vector1]))
    # 计算第二个向量的模
    sqrt_y = math.sqrt(sum([x ** 2 for x in vector2]))
    # 如果两个向量的模都为0，则返回0
    if sqrt_y == 0 or sqrt_y == 0:
        return 0
    # 返回两个向量的余弦相似度
    return x_dot_y / (sqrt_x * sqrt_y + 1e-7)


#输入一篇文本，寻找最相似文本
# 定义一个函数，用于搜索与给定段落最相似的文档
def search_most_similar_document(passage, corpus_vectors, vocab):
    # 将给定段落转换为向量
    input_vec = doc_to_vec(passage, vocab)
    # 初始化一个空列表，用于存储结果
    result = []
    # 遍历语料库中的每个向量
    for index, vector in enumerate(corpus_vectors):
        # 计算给定向量和当前向量的余弦相似度
        score = cosine_similarity(input_vec, vector)
        # 将结果添加到列表中
        result.append([index, score])
    # 按照相似度从高到低排序
    result = sorted(result, reverse=True, key=lambda x:x[1])
    # 返回前四个最相似的文档
    return result[:4]


if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, vocab, corpus = load_data(path)
    corpus_vectors = calculate_corpus_vectors(corpus, vocab)
    passage = "魔兽争霸"
    for corpus_index, score in search_most_similar_document(passage, corpus_vectors, vocab):
        print("相似文章:\n", corpus[corpus_index].strip())
        print("得分：", score)
        print("--------------")