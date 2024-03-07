import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns

# 数据加载
data = pd.read_csv('Credit_Card.csv')

print(data.shape) # 查看数据集大小
print(data.describe()) # 数据集概览

# 查看年龄分布情况
age = data['AGE']
payment = data[data["payment.next.month"]==1]['AGE']
bins =[20,30,40,50,60,70,80]
seg = pd.cut(age,bins,right=False)
print(seg)
counts =pd.value_counts(seg,sort=False)
b = plt.bar(counts.index.astype(str),counts)
plt.bar_label(b,counts)
plt.show()

#逾期的用户年龄分布
payment_seg = pd.cut(payment,bins,right=False)
counts1 =pd.value_counts(payment_seg,sort=False)
b2 = plt.bar(counts1.index.astype(str),counts1,color='r')
plt.bar_label(b2,counts1)
plt.show()

next_month = data['payment.next.month'].value_counts()
print(next_month)
df = pd.DataFrame({'payment.next.month': next_month.index,'values': next_month.values})
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure(figsize = (6,6))

plt.title('逾期率客户\n (还款：0,逾期：1)')
sns.set_color_codes("pastel")
sns.barplot(x = 'payment.next.month', y="values", data=df)
plt.show()

# 特征选择，去掉ID字段、最后一个结果字段即可
data.drop(['ID'], inplace=True, axis =1) #ID这个字段没有用
target = data['payment.next.month'].values
columns = data.columns.tolist()
columns.remove('payment.next.month')
features = data[columns].values

# 70%作为训练集，30%作为测试集
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify = target, random_state = 1)

# 构造各种分类器
classifiers = [
    SVC(random_state = 1, kernel = 'rbf'),     # 支持向量机分类
    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),  # 决策树分类
    RandomForestClassifier(random_state = 1, criterion = 'gini'),  # 随机森林分类
    KNeighborsClassifier(metric = 'minkowski'),    # K近邻分类
]
# 分类器名称
classifier_names = [
            'svc',
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
]
# 分类器参数
classifier_param_grid = [
            {'svc__C':[1], 'svc__gamma':[0.01]},
            {'decisiontreeclassifier__max_depth':[6,9,11]},
            {'randomforestclassifier__n_estimators':[3,5,6]} ,
            {'kneighborsclassifier__n_neighbors':[4,6,8]},
]

# 对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score)

    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" %search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率 %0.4lf" %accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response

for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')

