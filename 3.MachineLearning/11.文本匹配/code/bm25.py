import json
import math
import os
import pickle
import sys
from typing import Dict, List


class BM25:
    EPSILON = 0.25  # 用于处理负IDF值的经验常数
    PARAM_K1 = 1.5  # BM25算法中超参数 控制词频对得分的影响
    PARAM_B = 0.6  # BM25算法中超参数 控制文档长度对得分的影响

    def __init__(self, corpus: Dict):
        """
            初始化BM25模型
            :param corpus: 文档集, 文档集合应该是字典形式，key为文档的唯一标识，val对应其文本内容，文本内容需要分词成列表
        """

        self.corpus_size = 0  # 文档数量
        self.wordNumsOfAllDoc = 0  # 用于计算文档集合中平均每篇文档的词数 -> wordNumsOfAllDoc / corpus_size
        self.doc_freqs = {}  # 记录每篇文档中查询词的词频 {doc_id: {word: count}}
        self.idf = {}  # 记录查询词的 IDF {word: idf}
        self.doc_len = {}  # 记录每篇文档的单词数  {doc_id: length}
        self.docContainedWord = {}  # 包含单词 word 的文档集合  {word: set(doc_ids)}
        self._initialize(corpus)


    def _initialize(self, corpus: Dict):
        """
            根据语料库构建倒排索引
        """
        # nd = {} # word -> number of documents containing the word
        for index, document in corpus.items():
            self.corpus_size += 1
            self.doc_len[index] = len(document)  # 文档的单词数
            self.wordNumsOfAllDoc += len(document)

            frequencies = {}  # 一篇文档中单词出现的频率
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs[index] = frequencies

            # 构建词到文档的倒排索引，将包含单词与文档的包含关系进行反向映射
            # 示例：
            # docContainedWord = {
            # "apple": {"0", "1", "2"},  (文档序号)
            # "banana": {"1", "4"}
            # }
            for word in frequencies.keys():
                if word not in self.docContainedWord:
                    self.docContainedWord[word] = set()
                self.docContainedWord[word].add(index)

        # 计算 idf
        idf_sum = 0  # collect idf sum to calculate an average idf for epsilon value
        negative_idfs = []
        for word in self.docContainedWord.keys():
            doc_nums_contained_word = len(self.docContainedWord[word])
            #IDF(q)=log(N−包含q的文档数+0.5/包含q的文档数+0.5)
            idf = math.log(self.corpus_size - doc_nums_contained_word +
                           0.5) - math.log(doc_nums_contained_word + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        # 处理负IDF：用 EPSILON * 平均IDF 替代负值
        # 负IDF的物理意义：如果一个词出现在超过半数的文档中，BM25认为它“太常见”，对区分文档的贡献为负。
        average_idf = float(idf_sum) / len(self.idf)
        eps = BM25.EPSILON * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    @property
    def avgdl(self):
        return float(self.wordNumsOfAllDoc) / self.corpus_size


    def get_score(self, query: List, doc_index):
        """
        计算查询 q 和文档 d 的相关性分数
        :param query: 查询词列表
        :param doc_index: 为语料库中某篇文档对应的索引
        """
        k1 = BM25.PARAM_K1
        b = BM25.PARAM_B
        score = 0
        doc_freqs = self.doc_freqs[doc_index]  # 当前文档的词频字典
        for word in query:
            if word not in doc_freqs:
                continue
            # BM25公式核心部分
            score += self.idf[word] * doc_freqs[word] * (k1 + 1) / (
                    doc_freqs[word] + k1 * (1 - b + b * self.doc_len[doc_index] / self.avgdl))
        return [doc_index, score]

    def get_scores(self, query):
        scores = [self.get_score(query, index) for index in self.doc_len.keys()]
        return scores


