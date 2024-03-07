import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
#读取csv数据
data = pd.read_csv('weather.csv')

#将字符串编码为数字
label_encoder = LabelEncoder()
data['Outlook'] = label_encoder.fit_transform(data['Outlook'])
data['Temperature'] = label_encoder.fit_transform(data['Temperature'])
data['Humidity'] = label_encoder.fit_transform(data['Humidity'])
data['Windy'] = label_encoder.fit_transform(data['Windy'])

#将数字特征进行独热编码
one_hot_encoder = OneHotEncoder(categories='auto')
encoded_features = one_hot_encoder.fit_transform(data[['Outlook', 'Temperature']]).toarray()

#将Play列映射为二进制类别变量
data['Play'] = data['Play'].map({'Yes': 1, 'No': 0})

#将编码后的特征和标签分割
X = pd.concat([pd.DataFrame(encoded_features), data[['Windy', 'Humidity']]], axis=1)
y = data['Play']

#建立决策树模型并进行拟合
model = DecisionTreeClassifier()
model.fit(X, y)

#预测新数据
#【Sunny,Hot,High,Weak】编码为[0, 0, 1, 0, 1, 0, 1, 0]
new_data = pd.DataFrame([[0, 0, 1, 0, 1, 0, 1, 0]], columns=X.columns)
prediction = model.predict(new_data)

if prediction == 1:
    print('Play: No')
else:
    print('Play: Yes')

# 画出决策树
plt.figure(figsize=(8, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Play', 'Play'])
plt.show()