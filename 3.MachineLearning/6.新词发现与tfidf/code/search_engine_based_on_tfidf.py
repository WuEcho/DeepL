import jieba
import math
import os
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk
"""
基于tfidf实现简单搜索引擎
"""

jieba.initialize()

#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
def load_data(file_path):
    corpus = []
    with open(file_path, encoding="utf8") as f:
        documents = json.loads(f.read())
        for document in documents:
            corpus.append(document["title"] + "\n" + document["content"])
        tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict, corpus

# 定义一个搜索引擎函数，参数为查询词、tf-idf字典、语料库和返回结果数量
def search_engine(query, tf_idf_dict, corpus, top=3):
    # 使用jieba分词对查询词进行分词
    query_words = jieba.lcut(query)
    # 初始化结果列表
    res = []
    # 遍历tf-idf字典中的每个文档
    for doc_id, tf_idf in tf_idf_dict.items():
        # 初始化分数
        score = 0
        # 遍历查询词中的每个词
        for word in query_words:
            # 将查询词在文档中的tf-idf值累加到分数中
            score += tf_idf.get(word, 0)
        # 将文档id和分数添加到结果列表中
        res.append([doc_id, score])
    # 按照分数从高到低对结果列表进行排序
    res = sorted(res, reverse=True, key=lambda x:x[1])
    for i in range(top):
        doc_id = res[i][0]
        print(corpus[doc_id])
        print("--------------")
    return res

if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    while True:
        query = input("请输入您要搜索的内容:")
        search_engine(query, tf_idf_dict, corpus)