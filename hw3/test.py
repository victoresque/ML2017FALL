import numpy as np
import pandas as pd
import sys
from keras.models import load_model
from param import *

test_data_path = sys.argv[1]
prediction_path = sys.argv[2]

model = load_model('cnn.h5')

raw = pd.read_csv(test_data_path)
X_test = raw['feature']
X_test = np.array([[int(i) for i in x.split()] for x in X_test]).astype(np.float32)

X_test /= 255
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

y_test = model.predict(X_test)
y = np.zeros(y_test.shape[0])
for i in range(y_test.shape[0]):
    y[i] = np.argmax(y_test[i])

pd.DataFrame([[i, int(y[i])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv(prediction_path, index=False)