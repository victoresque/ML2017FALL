import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from util import *
from param import *

model = load_model('rnn_0.h5')

lines, labels = readData('data/training_label.txt')
dictionary = getDictionary(lines)

lines = readTestData('data/testing_data.txt')
lines = preprocessLines(lines)
lines = transformByDictionary(lines, dictionary)

x_test = sequence.pad_sequences(lines[:16], maxlen=maxlen)
y = model.predict(x_test, verbose=True).flatten()

y = np.array([int(i > 0.5) for i in y])
savePrediction(y, 'result/prediction.csv')