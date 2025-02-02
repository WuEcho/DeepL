import jieba
import math
import os
import random
import re
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk
"""
基于tfidf实现简单文本摘要
"""

jieba.initialize()

#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
# 定义一个函数，用于加载文件中的数据
def load_data(file_path):
    # 创建一个空列表，用于存储文档
    corpus = []
    # 打开文件，以utf8编码读取
    with open(file_path, encoding="utf8") as f:
        # 将文件中的数据加载为json格式
        documents = json.loads(f.read())
        # 遍历每个文档
        for document in documents:
            # 断言文档的标题和内容中不包含换行符
            assert "\n" not in document["title"]
            assert "\n" not in document["content"]
            # 将文档的标题和内容合并，并以换行符分隔，添加到corpus列表中
            corpus.append(document["title"] + "\n" + document["content"])
        # 调用calculate_tfidf函数，计算corpus中每个词的tf-idf值，并返回tf-idf字典和corpus列表
        tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict, corpus

#计算每一篇文章的摘要
#输入该文章的tf_idf词典，和文章内容
#top为人为定义的选取的句子数量
#过滤掉一些正文太短的文章，因为正文太短在做摘要意义不大
def generate_document_abstract(document_tf_idf, document, top=3):
    #将文档按照标点符号进行分割
    sentences = re.split("？|！|。", document)
    #过滤掉正文在五句以内的文章
    if len(sentences) <= 5:
        return None
    result = []
    for index, sentence in enumerate(sentences):
        sentence_score = 0
        #将句子进行分词
        words = jieba.lcut(sentence)
        for word in words:
            #计算句子中每个词的TF-IDF值，并累加
            sentence_score += document_tf_idf.get(word, 0)
        #计算句子得分
        sentence_score /= (len(words) + 1)
        result.append([sentence_score, index])
    #按照句子得分进行排序
    result = sorted(result, key=lambda x:x[0], reverse=True)
    #权重最高的可能依次是第10，第6，第3句，将他们调整为出现顺序比较合理，即3,6,10
    important_sentence_indexs = sorted([x[1] for x in result[:top]])
    #将重要的句子按照顺序拼接成摘要
    return "。".join([sentences[index] for index in important_sentence_indexs])

#生成所有文章的摘要
def generate_abstract(tf_idf_dict, corpus):
    # 定义一个空列表，用于存储生成的摘要
    res = []
    # 遍历tf_idf_dict字典中的每个元素
    for index, document_tf_idf in tf_idf_dict.items():
        # 获取corpus中对应索引的文档，并按照"\n"分割成标题和正文
        title, content = corpus[index].split("\n")
        # 调用generate_document_abstract函数，生成文档摘要
        abstract = generate_document_abstract(document_tf_idf, content)
        # 如果摘要为空，则跳过该文档
        if abstract is None:
            continue
        # 将生成的摘要添加到corpus中对应索引的文档后面
        corpus[index] += "\n" + abstract
        # 将标题、正文和摘要组成一个字典，添加到res列表中
        res.append({"标题":title, "正文":content, "摘要":abstract})
    # 返回res列表
    return res


if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    res = generate_abstract(tf_idf_dict, corpus)
    writer = open("abstract.json", "w", encoding="utf8")
    writer.write(json.dumps(res, ensure_ascii=False, indent=2))
    writer.close()
