import matplotlib.pyplot as plt
import numpy as np

confusion_matrix = np.array(
  [  [  0 ,   0   , 4   , 0    ,4 , 61  ,  0],
 [   0,    0,   14,    0,    8,   71,    0],
 [   0,    0,   28,    0,   28,  172,    0],
 [   0,    0,    0,    0,    0,   28,    0],
 [   0,    0,   15,    0,   17,  194,    0],
 [   0,    0,   13,    0,   11, 1314,    0],
 [   0,    0,    2,    0,    0,   19,    0]
 ])

# Plotting the confusion matrix
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['bkl', 'df', 'mel', 'vasc', 'bcc', 'nv', 'akiec']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted label')
plt.ylabel('True label')

# Displaying the values in the confusion matrix
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black')

plt.tight_layout()
plt.show()
