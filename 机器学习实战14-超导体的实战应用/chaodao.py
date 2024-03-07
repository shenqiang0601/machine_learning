import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf

# 加载数据
dataFrame = pd.read_csv('train.csv')

#doing a sanity check
# plt.style.use('ggplot')
# plt.figure(figsize = (50, 50))
# sb.heatmap(dataFrame.corr(), annot= True)
# plt.show()

# lets try to visualize range of tc, it's distribution and outliers
sb.displot(data = dataFrame, x = 'critical_temp')
sb.displot(data = dataFrame, x = 'critical_temp', kind = 'kde')

# plotting scatter plot for sample's Temperature
plt.style.use('ggplot')
plt.figure(figsize = (10, 7))
plt.scatter(dataFrame.index, dataFrame['critical_temp'], color = 'green')
plt.title('Scatter Plot for Sample Temperature')
plt.show()

featuresNeeded = dataFrame.columns[: -1]

features = np.array (dataFrame[featuresNeeded])
targets =  np.array(dataFrame['critical_temp'])

# let standardize our feature Vector First for PCA and futher usage
from sklearn.preprocessing import StandardScaler
stdc = StandardScaler()
features = stdc.fit_transform(features)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(features)
dimReducedFrame = pca.transform(features)

# converting to dataFrame and Plotting
dimReducedFrame = pd.DataFrame(dimReducedFrame)
dimReducedFrame = dimReducedFrame.rename(columns = {0: 'V1', 1 : 'V2'})
dimReducedFrame['critical_temp'] = targets

# Plotting
plt.figure(figsize = (10, 7))
sb.scatterplot(data = dimReducedFrame, x = 'V1', y = 'V2', hue = 'critical_temp')
plt.grid(True)
plt.show()

plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
sb.regplot(data = dimReducedFrame, x = 'V1', y = 'critical_temp' , color = 'green')
plt.title('Relation of Tc with 1st Principal Compnent')


plt.subplot(1, 2, 2)
sb.regplot(data = dimReducedFrame, x = 'V2', y = 'critical_temp' , color = 'green')
plt.title('Relation of Tc with 2nd  Principal Compnent')
plt.show()


# normalizing our targets
maxTc = max(targets)
targets = targets/1

# splitting data into train and test frame
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size = 0.2, random_state = 42)

# writting Function to visualize our models performance
from sklearn.metrics import mean_squared_error as mse
def PerformanceCalculator(trueVals, predVals, name):
    plt.plot([0,0.001,0.01,1], [0,0.001,0.01,1], color = 'blue')
    plt.scatter(trueVals, predVals, color = 'green')
    er = mse(trueVals, predVals)
    er = pow(er, 0.5)
    er = int(er * 10000) / 10000
    plt.title('RMSE: ' + str(er) + ' for '+ name)
    plt.show()

# lets start our Analysis with linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, 'Linear Regression')

# lets do one better by using a decission Tree
from sklearn.tree import DecisionTreeRegressor
lr = DecisionTreeRegressor()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, 'Decission Tree Regressor')

from sklearn.ensemble import GradientBoostingRegressor
lr = GradientBoostingRegressor()

lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, 'Gradient Boosting Regressor')


# lets try Random Forest
from sklearn.ensemble import RandomForestRegressor

lr1 = RandomForestRegressor()
lr1.fit(xTrain, yTrain)

predictions = lr1.predict(xTest)
PerformanceCalculator(yTest, predictions, 'Random Forest Regressor')

# Let's try bagging classifier
from sklearn.ensemble import BaggingRegressor
lr = BaggingRegressor()

#lr = RandomForestRegressor()
lr.fit(xTrain, yTrain)

predictions = lr.predict(xTest)
PerformanceCalculator(yTest, predictions, 'Bagging Regressor')
#

# lets try essembling
pred1 = lr1.predict(xTest)
pred2 = lr.predict(xTest)
predictions = (pred1 + pred2)/ 2

PerformanceCalculator(yTest, predictions, '(Random Forest + Bagging)')

ylab = yTest
predVals = lr1.predict(xTest)

diff = abs(ylab - predVals)
ansFrame = {
    'True Tc' :  ylab,
    'Predicted Tc': predVals,
    'Absolute error': diff
}

ansFrame = pd.DataFrame(ansFrame)
ansFrame[0:20]


