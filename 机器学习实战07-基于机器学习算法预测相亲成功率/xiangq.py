import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 读取数据
data = pd.read_csv('data.csv')

# 数据预处理
data.fillna(data.mean(), inplace=True)

# 划分训练集和测试集
X = data.drop('success', axis=1)
y = data['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

# 评估
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 输入样例数据进行预测
sample_data = pd.DataFrame({'house': [0],
                            'car': [0],
                            'appearance': [9],
                            'family_status': [5],
                            'parents_status': [5],
                            'lifestyle': [9],
                            'education': [5],
                            'personality': [8],
                            'interests': [8]})

sample_pred = rf.predict(sample_data)
print("Sample prediction:", "Success" if sample_pred[0] == 1 else "Failure")
