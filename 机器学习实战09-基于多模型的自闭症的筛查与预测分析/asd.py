import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('Autism_Data.arff',na_values='?')
df2 = pd.read_csv('Toddler Autism dataset July 2018.csv',na_values='?')

sns.set_style('whitegrid')
data1= df1[df1['Class/ASD']=='YES']
data2= df2[df2['Class/ASD Traits ']=='Yes']
print("Adults: ",len(data1)/len(df1) * 100)
print("Toddlers:",len(data2)/len(df2) * 100)

fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.heatmap(data1.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
ax[0].set_title('Adult dataset')
sns.heatmap(data2.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
ax[1].set_title('Toddlers dataset')
plt.show()

fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.countplot(x='jundice',data=data1,hue='gender',ax=ax[0])
ax[0].set_title('ASD positive Adults born with jaundice based on gender')
ax[0].set_xlabel('Jaundice while birth')
sns.countplot(x='Jaundice',data=data2,hue='Sex',ax=ax[1])
ax[1].set_title('ASD positive Toddlers born with jaundice based on gender')
ax[1].set_xlabel('Jaundice while birth')
plt.show()

#Let's see the age distribution of ASD positive
fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.distplot(data1['age'],kde=False,bins=45,color='darkred',ax=ax[0])
ax[0].set_xlabel('Adult age in years')
ax[0].set_title('Age distribution of ASD positive')
sns.distplot(data2['Age_Mons'],kde=False,bins=30,color='darkred',ax=ax[1])
ax[1].set_xlabel('Toddlers age in months')
ax[1].set_title('Age distribution of ASD positive')
plt.show()

plt.figure(figsize=(20,6))
sns.countplot(x='contry_of_res',data=data1,order= data1['contry_of_res'].value_counts().index[:15],hue='gender',palette='viridis')
plt.title('Positive ASD Adults country wise distribution')
plt.xlabel('Countries')
plt.tight_layout()

#Lets see the ethnicity value counts
print(data1['ethnicity'].value_counts())
data2['Ethnicity'].value_counts()

plt.figure(figsize=(15,6))
sns.countplot(x='contry_of_res',data=data1[data1['ethnicity']=='White-European'],order=data1[data1['ethnicity']=='White-European']['contry_of_res'].value_counts().index[:10],palette='viridis')
plt.title('Positive ASD of White and European Ethnicities country wise distribution')
plt.xlabel('Countries')
plt.tight_layout()

fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.countplot(x='austim',data=data1,hue='ethnicity',palette='rainbow',ax=ax[0])
ax[0].set_title('Positive ASD Adult relatives with Autism distribution for different ethnicities')
ax[0].set_xlabel('Adult Relatives with ASD')
sns.countplot(x='Family_mem_with_ASD',data=data2,hue='Ethnicity',palette='rainbow',ax=ax[1])
ax[1].set_title('Positive ASD Toddler relatives with Autism distribution for different ethnicities')
ax[1].set_xlabel('Toddler Relatives with ASD')
plt.tight_layout()

within24_36= pd.get_dummies(df2['Age_Mons']>24,drop_first=True)
within0_12 = pd.get_dummies(df2['Age_Mons']<13,drop_first=True)
male=pd.get_dummies(df2['Sex'],drop_first=True)
ethnics=pd.get_dummies(df2['Ethnicity'],drop_first=True)
jaundice=pd.get_dummies(df2['Jaundice'],drop_first=True)
ASD_genes=pd.get_dummies(df2['Family_mem_with_ASD'],drop_first=True)
ASD_traits=pd.get_dummies(df2['Class/ASD Traits '],drop_first=True)

final_data= pd.concat([within0_12,within24_36,male,ethnics,jaundice,ASD_genes,ASD_traits],axis=1)
final_data.columns=['within0_12','within24_36','male','Latino','Native Indian','Others','Pacifica','White European','asian','black','middle eastern','mixed','south asian','jaundice','ASD_genes','ASD_traits']
final_data.head()

from sklearn.model_selection import train_test_split
X= final_data.iloc[:,:-1]
y= final_data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01,0.1,1,10,100,1000]}
grid_log = GridSearchCV(LogisticRegression(),param_grid,refit=True)
grid_log.fit(X_train,y_train)
print('GridSearchCV')
print(grid_log.best_estimator_)
pred_log=grid_log.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))

from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
pred_rfc= rfc.predict(X_test)
print('RandomForestClassifier')
print(confusion_matrix(y_test,pred_rfc))
print(classification_report(y_test,pred_rfc))

#knn
#first scale the variables
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)
X_scaled = pd.DataFrame(scaled_features,columns=X.columns)
X_scaled.head()
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
error_rate =[]

for i in range (1,50):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn= KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)

pred_knn=knn.predict(X_test)
print(confusion_matrix(y_test,pred_knn))
print(classification_report(y_test,pred_knn))