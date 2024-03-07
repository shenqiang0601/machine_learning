import pandas as pd
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split


# 6.2 数据加载与清洗
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 这里我们假设CSV文件中的数据已经经过初步清洗
    return df


# 6.3 推荐模型构建
def build_model(data):
    # 使用Surprise库中的KNNWithMeans算法
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
    algo.fit(trainset)
    return algo


# 6.4 推荐结果展示与评估
def recommend_movies(model, data, user_id, num_recommendations=5):
    user_ratings = data[data['user_id'] == user_id]
    user_ratings = user_ratings.merge(data[['movie_id', 'title']], on='movie_id')
    user_ratings.set_index('title', inplace=True)

    # 获取用户未评分的电影
    unseen_movies = data[data['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id', 'title']]

    # 预测用户对未评分电影的评分
    predictions = model.test(user_ratings.index.values, unseen_movies.index.values)

    # 对预测结果进行排序并获取前N个推荐
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    recommendations = [data[data['movie_id'] == movie_inner_id]['title'].iloc[0] for movie_inner_id, _ in top_n]

    return recommendations


# 主函数
def main():
    file_path = 'movie_data2.csv'  # 假设CSV文件名为movie_data.csv
    df = load_data(file_path)

    # 构建模型
    model = build_model(df)

    # 为用户推荐电影
    user_id = 1  # 假设我们要为用户ID为1的用户推荐电影
    recommendations = recommend_movies(model, df, user_id)

    print(f"为用户 {user_id} 推荐的电影:")
    for movie in recommendations:
        print(movie)


if __name__ == "__main__":
    main()
