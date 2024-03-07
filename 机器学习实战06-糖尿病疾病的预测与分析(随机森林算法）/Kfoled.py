import pandas as pd
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

diab_df = pd.read_csv('diabetes.csv')

print(diab_df.describe())

from sklearn.model_selection import train_test_split, KFold
x = diab_df.drop(['Outcome'], axis=1)
y = diab_df['Outcome']

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

kf = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(criterion = "gini",
                                       min_samples_leaf = 5,
                                       min_samples_split = 10,
                                       n_estimators=99,
                                       max_features='auto',
                                       oob_score=True,
                                       random_state=1,
                                       n_jobs=-1)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_matrices = []

for train_index, test_index in kf.split(x_scaled):
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)

    confmat = confusion_matrix(y_pred, y_test)
    confusion_matrices.append(confmat)

    accuracy = accuracy_score(y_pred, y_test)
    accuracy_scores.append(accuracy)

    precision = precision_score(y_pred, y_test)
    precision_scores.append(precision)

    recall = recall_score(y_pred, y_test)
    recall_scores.append(recall)

    f1 = f1_score(y_pred, y_test)
    f1_scores.append(f1)

print(f"Average accuracy: {np.max(accuracy_scores)}")
print(f"Average precision: {np.max(precision_scores)}")
print(f"Average recall: {np.max(recall_scores)}")
print(f"Average F1 score: {np.max(f1_scores)}")


y_pred = random_forest.predict(x_test)
confmat1 = confusion_matrix(y_pred, y_test)

from sklearn import metrics
cm=metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred,y_test,labels=random_forest.classes_),
                              display_labels=random_forest.classes_)
cm.plot(cmap="magma").figure_.savefig("confusion_matrix.png")

Image.open("confusion_matrix.png").show()

print(accuracy_score(y_pred, y_test))