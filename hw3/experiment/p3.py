import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from util import *
from param import *


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] < 0.2 or cm[i, j]> 0.7 else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

X_train, y_train = train_data('../data/X_train.npy', '../data/y_train.npy')
X_train, y_train, X_test, y_test = split_valid(X_train, y_train)
model = load_model('../model/0.67623.h5')

predictions = model.predict(X_test)
y = np.zeros(predictions.shape[0])
for i in range(predictions.shape[0]):
    y[i] = np.argmax(predictions[i])
predictions = y
y = np.zeros(y_test.shape[0])
for i in range(y_test.shape[0]):
    y[i] = np.argmax(y_test[i])
y_test = y

print(y_test.shape, predictions.shape)
conf_mat = confusion_matrix(y_test, predictions)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()