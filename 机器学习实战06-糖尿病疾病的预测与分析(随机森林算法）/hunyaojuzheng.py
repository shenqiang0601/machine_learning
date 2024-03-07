import matplotlib.pyplot as plt
import numpy as np
import itertools

confusion_matrix = np.array([[96, 14], [11, 33]])

plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
tick_marks = np.arange(len(confusion_matrix))
class_labels = ['Class 0', 'Class 1']
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if confusion_matrix[i, j] > thresh else "black")

plt.savefig("confusion_matrix11111.png")
plt.show()