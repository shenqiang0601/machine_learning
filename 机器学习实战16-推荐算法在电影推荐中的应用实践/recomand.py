import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 读取电影数据
data = pd.read_csv('movie_data.csv')

# 构建电影-用户矩阵
movie_user_matrix = data.pivot(index='movie_id', columns='user_id', values='rating')

# 填充缺失值
movie_user_matrix.fillna(0, inplace=True)

# 计算电影之间的相似度
movie_similarity = cosine_similarity(movie_user_matrix)

# 输出电影相似度矩阵
print(movie_similarity)

# 为用户推荐电影
def recommend_movies(user_id, movie_similarity, data, num_recommendations=4):
    # 找到用户评分过的电影
    user_rated_movies = data[data['user_id'] == user_id]['movie_id']

    # 计算用户评分过的电影与其他电影的相似度
    similarity_scores = []
    for movie_id in user_rated_movies:
        similarity_scores.extend(list(enumerate(movie_similarity[movie_id - 1])))

    # 对相似度进行排序
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(similarity_scores)

    # 获取推荐电影ID
    recommended_movie_ids = [movie_id for movie_id, score in similarity_scores[:num_recommendations]]

    return recommended_movie_ids

# 为用户1推荐电影
recommended_movie_ids = recommend_movies(1, movie_similarity, data)
print(recommended_movie_ids)
