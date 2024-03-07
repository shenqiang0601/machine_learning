from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# 读入CSV文件
df = pd.read_csv('test.csv')

# 创建TF-IDF特征矩阵
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['text'])

# 对特征矩阵进行层次聚类
cluster = AgglomerativeClustering(distance_threshold=0.8, n_clusters=None)
cluster_labels = cluster.fit_predict(features.toarray())

# 保存聚类结果到DataFrame中
df_cluster = pd.DataFrame({'cluster_label': cluster_labels, 'text': df['text']})

# 用groupby对聚类结果进行分组
grouped_clusters = df_cluster.groupby('cluster_label')

# 标注相似度大于阀值的样本
similar_sentences = []
threshold = 0.5
for group_name, group_df in grouped_clusters:
    if len(group_df) > 1:
        # 计算每一对之间的相似度
        similarity_matrix = features[group_df.index] * features[group_df.index].T
        np.fill_diagonal(similarity_matrix.A, -1)
        similar_pairs = [(group_df.index[i], group_df.index[j])
                         for i, j in zip(*np.where(similarity_matrix.toarray() > threshold))]
        similar_sentences.extend(similar_pairs)

# 保存标注结果到DataFrame中
df_similar = pd.DataFrame({'text': df.loc[[idx for pair in similar_sentences for idx in pair], 'text'],
                           'is_similar': True})
df_similar = df_similar.groupby('text').any().reset_index()

# 将标注结果合并到原始DataFrame中
df = pd.merge(df, df_similar, how='left', on='text')

# 将结果保存到CSV文件中
df.to_csv('output.csv', index=False)