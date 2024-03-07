import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 假设df是已经读取的CSV数据
df = pd.DataFrame({
    'user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    'movie_id': [1, 2, 3, 4, 1, 2, 3, 5, 1, 6, 7],
    'rating': [5, 4, 3, 2, 4, 5, 2, 4, 5, 4, 3],
    'title': ["The Shawshank Redemption", "The Godfather", "Pulp Fiction", "Forrest Gump",
              "The Shawshank Redemption", "The Godfather", "Pulp Fiction", "The Dark Knight",
              "The Shawshank Redemption", "Inception", "The Matrix"],
    'genre': ["Drama", "Crime", "Crime", "Drama", "Drama", "Crime", "Crime", "Action", "Drama", "Sci-Fi", "Sci-Fi"],
    'release_year': [1994, 1972, 1994, 1994, 1994, 1972, 1994, 2008, 1994, 2010, 1999]
})
print(df )

# 读取电影数据
# data = pd.read_csv('movie_data2.csv')
# df = data
# print(df )

# 构建一个用户-电影评分矩阵
user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

# 填充缺失值为0
user_movie_matrix = user_movie_matrix.fillna(0)

# 计算余弦相似度
user_similarity = cosine_similarity(user_movie_matrix)

# 转换为DataFrame以便查看
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# 找到与用户1最相似的用户
most_similar_users = user_similarity_df.loc[1].sort_values(ascending=False).index[1:]  # 排除用户自己

# 找出这些用户评价较高的电影
recommended_movies = df[(df['user_id'].isin(most_similar_users)) &
                        ~(df['movie_id'].isin(df[df['user_id'] == 1]['movie_id']))].groupby('movie_id').mean().sort_values(by='rating', ascending=False)

print(recommended_movies)

