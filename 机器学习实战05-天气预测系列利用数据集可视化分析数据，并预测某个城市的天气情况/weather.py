import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']

df = pd.read_csv('weather_dataset.csv')
labels = pd.read_csv('weather_labels.csv')

df_budapest = pd.concat([df.iloc[:,:2],df.iloc[:,11:19]],axis=1)
df_budapest['DATE'] = pd.to_datetime(df_budapest['DATE'],format='%Y%m%d')

# Function to count mean values for the features
def mean_for_mth(feature):
    mean = []
    for x in range(12):
        mean.append(
            float("{:.2f}".format(df_budapest[df_budapest['MONTH'] == (x+1)][feature].mean())))
    return mean

df_budapest.drop(['MONTH'],axis=1).describe()

#sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
plt.plot(df_budapest['DATE'][:365],df_budapest['BUDAPEST_temp_mean'][:365])
plt.title('2000年的温度变化图')
plt.xlabel('DATE')
plt.ylabel('DEGREE')

plt.show()


months = ['Jan', 'Febr', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
          'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
mean_temp = mean_for_mth('BUDAPEST_temp_mean')

plt.figure(figsize=(12,6))
bar = plt.bar(x = months,height = mean_temp, width = 0.8, color=['thistle','mediumaquamarine',                'orange'])
plt.xticks(rotation = 45)
plt.xlabel('MONTHS')
plt.ylabel('DEGREES')
plt.title('布达佩斯(匈牙利首都)年平均温度')
plt.bar_label(bar)
plt.show()

mean_temp = mean_for_mth('BUDAPEST_humidity')
plt.figure(figsize=(12,6))
bar = plt.bar(x = months, height = mean_temp, width = 0.8, color=['thistle','mediumaquamarine',                'orange'])
plt.xticks(rotation = 45)
plt.xlabel('MONTHS')
plt.ylabel('HUMIDITY')
plt.title('布达佩斯(匈牙利首都)的年平均湿度')
plt.bar_label(bar)
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(12,8))
fig.suptitle('各指标的分布次数图')
sns.histplot(data = df_budapest, x ='BUDAPEST_pressure', ax=axs[0,0], color='red', kde=True)
sns.histplot(data = df_budapest, x ='BUDAPEST_humidity', ax=axs[0,1], color='orange', kde=True)
sns.histplot(data = df_budapest, x ='BUDAPEST_temp_mean', ax=axs[1,0], kde=True)
sns.histplot(data = df_budapest, x ='BUDAPEST_global_radiation', ax=axs[1,1], color='green', kde=True)
plt.show()

labels_budapest = labels['BUDAPEST_BBQ_weather']
sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
sns.countplot(x = labels_budapest).set(title='Labels for BUDAPEST')

true_val = len(labels_budapest[labels_budapest == True])
false_val = len(labels_budapest[labels_budapest == False])
print('Precent of True values: {0:.2f}%'.format(true_val/(true_val+false_val)*100))
print('Precent of False values: {0:.2f}%'.format(false_val/(true_val+false_val)*100))


labels_budapest = labels_budapest.astype(int)
plt.figure(figsize=(12,6))
sns.set(style="darkgrid")
sns.boxplot(y = df_budapest['BUDAPEST_temp_mean'], x = labels_budapest).set(title='Relation between the temperature and the bbq weather')

labels_budapest = labels_budapest.astype(int)
df_budapest = df_budapest.drop(['DATE'],axis=1)
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler()
ovrspl_X, ovrspl_y  = oversample.fit_resample(df_budapest, labels_budapest)

labels_budapest = labels['BUDAPEST_BBQ_weather']
sns.set(style="darkgrid")
plt.figure(figsize=(12,6))
sns.countplot(x = ovrspl_y).set(title='Oversampled Labels for BUDAPEST')

true_val = len(ovrspl_y[ovrspl_y == 1])
false_val = len(ovrspl_y[ovrspl_y == 0])
print('Precent of True values: {0:.1f}%'.format(true_val/(true_val+false_val)*100))
print('Precent of False values: {0:.1f}%'.format(false_val/(true_val+false_val)*100))

plt.figure(figsize=(12,6))
sns.set(style="darkgrid")
sns.heatmap(df_budapest.corr(),annot=True,cmap='coolwarm').set(title='Correlation between features')
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
norm_X = scaler.fit_transform(ovrspl_X)
norm_X = pd.DataFrame(norm_X, columns=df_budapest.columns)
norm_X.describe()

from sklearn.model_selection import train_test_split
# Splitting dataset on training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(ovrspl_X,ovrspl_y, test_size = 0.3, random_state = 42)
print('Training set: ' + str(len(X_train)))
print('Testing set: ' + str(len(X_test)))

from sklearn.svm import SVC

model = SVC(verbose=True, kernel = 'linear', random_state = 0)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Classification report--------------------------------')
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict), annot=True, fmt='g').set(title='Confusion Matrix')

print('Model accuracy is: {0:.2f}%'.format(accuracy_score(y_test, y_predict)*100))
