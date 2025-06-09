import numpy as np
import pandas as pd
import pickle
import os
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
"""
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
"""

# 定义分词函数
def split_text(text):
    """
    对文本进行分词处理
    :param text: 输入文本
    :return: 分词后的文本，以空格分隔
    """
    # 使用正则表达式去除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba进行分词
    words = jieba.cut(text)
    # 返回分词结果，以空格分隔
    return ' '.join(words)

path = './'
# 数据加载
news = pd.read_csv(path+'sqlResult.csv',encoding='gb18030')
# 处理缺失值
print(news[news.content.isna()].head(5))
news=news.dropna(subset=['content'])

# 加载清洗后的corpus
if not os.path.exists(path+"corpus.pkl"):
    # 对所有文本进行分词
    corpus=list(map(split_text,[str(i) for i in news.content]))
    print(corpus[0])
    print(len(corpus))
    print(corpus[1])
    # 保存到文件，方便下次调用
    with open(path+'corpus.pkl','wb') as file:
        pickle.dump(corpus, file)
else:
    # 调用上次处理的结果
    with open(path+'corpus.pkl','rb') as file:
        corpus = pickle.load(file)

# 得到corpus的TF-IDF矩阵
if not os.path.exists(path+"tfidf.pkl"):
    countvectorizer = CountVectorizer(encoding='gb18030',min_df=0.015)
    tfidftransformer = TfidfTransformer()
    countvector = countvectorizer.fit_transform(corpus)
    print(countvector.shape)
    tfidf = tfidftransformer.fit_transform(countvector)
    print(tfidf.shape)

    # 保存到文件，方便下次调用
    with open(path+'tfidf.pkl','wb') as file:
        pickle.dump(tfidf, file)
else:
    # 调用上次处理的结果
    with open(path+'tfidf.pkl','rb') as file:
        tfidf = pickle.load(file)

#print(type(tfidf))
# 将csr_matrix 转换为 numpy.ndarray类型, 同时将原来float64类型转换为float32类型
tfidf = tfidf.toarray().astype(np.float32)
# embedding的维度
d = tfidf.shape[1]
print(d)
print(tfidf.shape)
print(type(tfidf))
#print(tfidf[1])
print(type(tfidf[1][1]))


# 精确索引
import faiss
index = faiss.IndexFlatL2(d)  # 构建 IndexFlatL2
print(index.is_trained)  # False时需要train
index.add(tfidf)  #添加数据
print(index.ntotal)  #index中向量的个数

#精确索引无需训练便可直接查询
k = 10  # 返回结果个数
cpindex = 3352
query_self = tfidf[cpindex:cpindex+1]  # 查询本身
dis, ind = index.search(query_self, k)
print(dis.shape) # 打印张量 (5, 10)
print(ind.shape) # 打印张量 (5, 10)
print(dis)  # 升序返回每个查询向量的距离
print(ind)  # 升序返回每个查询向量

print('怀疑抄袭:\n', news.iloc[cpindex].content)
# 找一篇相似的原文
similar2 = ind[0][1]
print(similar2)
print('相似原文:\n', news.iloc[similar2].content)
