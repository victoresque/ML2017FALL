import numpy as np
import matplotlib.pyplot as plt
from param import *
from util import *

id = 987
X_train, y_train = train_data('../data/X_train.npy', '../data/y_train.npy')
X_train, y_train, X_test, y_test = split_valid(X_train, y_train)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)

for i in range(len(y_test)):
    if np.argmax(y_test[i]) == 4:
        img = X_test[i]
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()