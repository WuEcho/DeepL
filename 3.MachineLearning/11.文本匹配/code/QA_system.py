import os
import json
import jieba
import numpy as np
from bm25 import BM25
from similarity_function import editing_distance, jaccard_distance
from gensim.models import Word2Vec

'''
基于faq知识库和文本匹配算法进行意图识别，完成单轮问答
'''

class QASystem:
    def __init__(self, know_base_path, algo):
        '''
        :param know_base_path: 知识库文件路径
        :param algo: 选择不同的算法
        '''
        self.load_know_base(know_base_path)
        self.algo = algo
        if algo == "bm25":
            self.load_bm25()
        elif algo == "word2vec":
            self.load_word2vec()
        else:
            #其余的算法不需要做事先计算
            pass

    def load_bm25(self):
        self.corpus = {}
        for target, questions in self.target_to_questions.items():
            self.corpus[target] = []
            for question in questions:
                self.corpus[target] += jieba.lcut(question)
        self.bm25_model = BM25(self.corpus)

    #词向量的训练
    def load_word2vec(self):
        #词向量的训练需要一定时间，如果之前训练过，我们就直接读取训练好的模型
        #注意如果数据集更换了，应当重新训练
        #当然，也可以收集一份大量的通用的语料，训练一个通用词向量模型。一般少量数据来训练效果不会太理想
        if os.path.isfile("model.w2v"):
            self.w2v_model = Word2Vec.load("model.w2v")
        else:
            #训练语料的准备，把所有问题分词后连在一起
            corpus = []
            for questions in self.target_to_questions.values():
                for question in questions:
                    corpus.append(jieba.lcut(question))
            #调用第三方库训练模型
            self.w2v_model = Word2Vec(corpus, vector_size=100, min_count=1)
            #保存模型
            self.w2v_model.save("model.w2v")
        #借助词向量模型，将知识库中的问题向量化
        self.target_to_vectors = {}
        for target, questions in self.target_to_questions.items():
            vectors = []
            for question in questions:
                vectors.append(self.sentence_to_vec(question))
            self.target_to_vectors[target] = np.array(vectors)

    # 将文本向量化
    def sentence_to_vec(self, sentence):
         #这里的初始化有几个作用 1：零初始化确保初始向量与模型的维度一致，避免后续操作出现维度错误。
         # 2.若直接取第一个词的向量初始化，当第一个词不存在于模型时，vector会未定义，导致错误。
        vector = np.zeros(self.w2v_model.vector_size) 
        words = jieba.lcut(sentence)
        # 所有词的向量相加求平均，作为句子向量
        count = 0
        for word in words:
            if word in self.w2v_model.wv:
                count += 1
                vector += self.w2v_model.wv[word]
        vector = np.array(vector) / count
        #文本向量做l2归一化，方便计算cos距离
        # ​L2归一化：此操作将向量除以其L2范数（即向量各元素平方和的平方根），使向量变为单位向量（长度为1）。
        # ​简化相似度计算：余弦相似度的计算公式为两向量的点积除以L2范数的乘积。归一化后，点积直接等于余弦相似度，计算更高效。
        #​ 数值稳定性：归一化可避免长向量在计算中占据过大权重，提升模型鲁棒性
        vector = vector / np.sqrt(np.sum(np.square(vector)))
        return vector

    def load_know_base(self, know_base_path):
        self.target_to_questions = {}
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    def query(self, user_query):
        results = []
        if self.algo == "editing_distance":
            for target, questions in self.target_to_questions.items():
                scores = [editing_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "jaccard_distance":
            for target, questions in self.target_to_questions.items():
                scores = [jaccard_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "bm25":
            words = jieba.lcut(user_query)
            results = self.bm25_model.get_scores(words)
        elif self.algo == "word2vec":
            #这个地方需要注意一下
            query_vector = self.sentence_to_vec(user_query)
            for target, vectors in self.target_to_vectors.items():
                # 如果向量已经归一化（长度为1），​点积（dot product）等于余弦相似度。
                # vectors 是一个二维数组，形状为 (n_questions, vector_size)（例如 10 个问题，每个问题向量是 100 维）。
                # vectors.transpose() 转置后形状为 (vector_size, n_questions)。
                # query_vector 的形状是 (vector_size,)。
                # query_vector.dot(vectors.transpose()) 的结果是一个一维数组，形状为 (n_questions,)，表示用户查询与目标下每个问题的相似度。
                cos = query_vector.dot(vectors.transpose())
                # print(cos)
                # 对目标下所有问题的相似度取平均值（np.mean(cos)），代表该目标与用户查询的整体匹配程度
                results.append([target, np.mean(cos)])
        else:
            assert "unknown algorithm!!"
        # 这段代码的作用是将列表 results 中的元素按照每个元素的第二个值（通常是分数或相似度）从高到低进行排序。具体解释如下：
        # ​**sorted(results, ...)**
        # 使用 Python 内置的 sorted() 函数对列表 results 进行排序，返回一个新的排序后的列表，原列表 results 保持不变。
        # ​**key=lambda x: x[1]**
        #l ambda x: x[1] 是一个匿名函数，表示取每个元素（x）的第二个值（索引为 1）作为排序依据。
        # 假设 results 的每个元素是形如 [目标, 分数] 的列表或元组（例如 ["流量套餐", 0.8]），则 x[1] 对应分数值。
        ​# **reverse=True**
        # 表示降序排序（从大到小）。分数高的元素会排在前面。
        sort_results = sorted(results, key=lambda x:x[1], reverse=True)
        return sort_results[:3]


if __name__ == '__main__':
    qas = QASystem("data/train.json", "bm25")
    question = "话费是否包月超了"
    res = qas.query(question)
    print(question)
    print(res)
    #
    # while True:
    #     question = input("请输入问题：")
    #     res = qas.query(question)
    #     print("命中问题：", res)
    #     print("-----------")

