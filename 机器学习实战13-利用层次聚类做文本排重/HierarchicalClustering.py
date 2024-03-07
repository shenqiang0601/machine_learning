# 必要的库
import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

# 设置常量
MAX_DF = 0.8 # 移除文档频率高于max_df的词语
MIN_DF = 0.01 # 将出现在少于min_df文档中的词语移除
STOP_WORDS = [line.strip() for line in open('stop_words.txt', encoding = 'utf-8').readlines()]

# 数据载入
data = pd.read_csv('test.csv', encoding = 'utf-8')
texts = data['text'].tolist()

# 分词
texts_fenci = []
for text in texts:
    words = jieba.cut(text)
    text_fenci = ''
    for word in words:
        if word not in STOP_WORDS:
            text_fenci += word + ' '
    texts_fenci.append(text_fenci)

# 文本向量化转换
vectorizer = TfidfVectorizer(max_df = MAX_DF, min_df = MIN_DF)
tf_idf = vectorizer.fit_transform(texts_fenci)

# 计算余弦距离
tf_idf_norm = tf_idf.toarray() / np.linalg.norm(tf_idf.toarray(), axis = -1).reshape(-1, 1)
cos_dist = cosine_similarity(tf_idf_norm)
print(cos_dist)
# 层次聚类
linkage_matrix = linkage(cos_dist, method='ward', metric='euclidean')
clusters = fcluster(linkage_matrix,t=0.2,criterion='distance')
print(clusters)
# 相似句子标注
cluster_dict = defaultdict(list)
for i, label in enumerate(clusters):
    cluster_dict[label].append(i)
for _ , idx in cluster_dict.items():

    if len(idx) > 1:
        print('相似句子为:', end = ' ')
        for i in idx:
            print(texts[i], end = ' ')
        print('\n')
