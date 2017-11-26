import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from util import *
from param import *

model = load_model('rnn_0.h5')

lines, labels = readData('data/training_label.txt')
lines, dictionary = getDictionaryAndTransform(lines)
print(len(lines))

lines = readTestData('data/testing_data.txt')
lines = transformByDictionary(lines, dictionary)
print(len(lines))

x_test = lines[:16]
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

y = model.predict(x_test, verbose=True).flatten()
print(y)
y = np.array([int(i>0.5) for i in y])
print(y)
pd.DataFrame([[i, int(y[i])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv('result/prediction.csv', index=False)