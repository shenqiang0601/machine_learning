import pandas as pd
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

diab_df= pd.read_csv('diabetes.csv')

print(diab_df.describe())

from sklearn.model_selection import train_test_split
x = diab_df.drop(['Outcome'],axis=1)
y = diab_df['Outcome']

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
x_scaled= sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(criterion = "gini",
                                       min_samples_leaf = 5,
                                       min_samples_split = 10,
                                       n_estimators=99,
                                       max_features='auto',
                                       oob_score=True,
                                       random_state=1,
                                       n_jobs=-1)

random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

confmat1 = confusion_matrix(y_pred, y_test)

from sklearn import metrics
cm=metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred,y_test,labels=random_forest.classes_),
                              display_labels=random_forest.classes_)
cm.plot(cmap="magma").figure_.savefig("confusion_matrix.png")

Image.open("confusion_matrix.png").show()

print(accuracy_score(y_pred, y_test))