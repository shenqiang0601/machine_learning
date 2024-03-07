import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor

# 加载数据
dataFrame = pd.read_csv('train.csv')

featuresNeeded = dataFrame.columns[:-1]

features = np.array(dataFrame[featuresNeeded])
targets = np.array(dataFrame['critical_temp'])

# 对特征向量进行标准化，以便进行PCA和进一步处理
stdc = StandardScaler()
features = stdc.fit_transform(features)

pca = PCA(n_components=2)
pca.fit(features)
dimReducedFrame = pca.transform(features)

# 转换为DataFrame并绘图
dimReducedFrame = pd.DataFrame(dimReducedFrame)
dimReducedFrame = dimReducedFrame.rename(columns={0: 'V1', 1: 'V2'})
dimReducedFrame['critical_temp'] = targets

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# 绘制散点图
plt.figure(figsize=(10, 7))
sb.scatterplot(data=dimReducedFrame, x='V1', y='V2', hue='critical_temp')
plt.grid(True)
plt.show()

# 绘制第一主成分与Tc之间的关系图
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sb.regplot(data=dimReducedFrame, x='V1', y='critical_temp', color='blue')
plt.title('Tc与第一主成分的关系')

# 绘制第二主成分与Tc之间的关系图
plt.subplot(1, 2, 2)
sb.regplot(data=dimReducedFrame, x='V2', y='critical_temp', color='blue')
plt.title('Tc与第二主成分的关系')
plt.show()

# 对目标值进行归一化
maxTc = max(targets)
targets = targets / 1

# 将数据拆分为训练集和测试集
xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size=0.2, random_state=42)

# 定义用于评估模型性能的函数
def PerformanceCalculator(trueVals, predVals, name):
    plt.plot([0, 0.001, 0.01, 1], [0, 0.001, 0.01, 1], color='blue')
    plt.scatter(trueVals, predVals, color='red')
    er = mse(trueVals, predVals)
    er = pow(er, 0.5)
    er = int(er * 10000) / 10000
    plt.title('RMSE: ' + str(er) + ' for ' + name)
    plt.show()

# 使用线性回归进行分析
lr = LinearRegression()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, '线性回归')

# 使用决策树进行分析
lr = DecisionTreeRegressor()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, '决策树回归')

# 使用梯度提升回归进行分析
lr = GradientBoostingRegressor()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, '梯度提升回归')

# 使用随机森林进行分析
lr1 = RandomForestRegressor()
lr1.fit(xTrain, yTrain)

predictions = lr1.predict(xTest)
PerformanceCalculator(yTest, predictions, '随机森林回归')

# 使用Bagging回归进行分析
lr = BaggingRegressor()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, 'Bagging回归')
#

# 使用集成方法进行预测
pred1 = lr1.predict(xTest)
pred2 = lr.predict(xTest)
predictions = (pred1 + pred2)/ 2

# 评估集成模型性能
PerformanceCalculator(yTest, predictions, '(随机森林 + Bagging)')

# 绘制真实值与预测值的关系图
ylab = yTest
predVals = lr1.predict(xTest)

plt.plot([0, 0.001, 0.01, 1], [0, 0.001, 0.01, 1], color='blue')
plt.scatter(ylab, predVals, color='green')
er = mse(ylab, predVals)
er = pow(er, 0.5)
er = int(er * 10000) / 10000
plt.title('真实值与预测值的关系图 (RMSE: ' + str(er) + ')')
plt.show()

