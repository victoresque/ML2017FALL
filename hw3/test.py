import numpy as np
import pandas as pd
import keras
from keras.models import load_model

model = load_model('model/cnn.h5')

num_classes = 7
img_rows, img_cols = 48, 48

X_test = np.load('data/X_test.npy')

X_test /= 255
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

y_test = model.predict(X_test)
y = np.zeros(y_test.shape[0])
for i in range(y_test.shape[0]):
    y[i] = np.argmax(y_test[i])

pd.DataFrame([[i, int(y[i])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv('result/prediction.csv', index=False)