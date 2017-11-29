import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

model = load_model('rnn_simple.h5')

lines, labels = readData('data/training_label.txt')
w2v = Word2Vec.load('data/word2vec.mdl')

lines = readTestData('data/testing_data.txt')
lines = preprocessLines(lines)
lines = transformByDictionary(lines, dictionary)

x_test = sequence.pad_sequences(lines, maxlen=maxlen)
y = model.predict(x_test, verbose=True).flatten()
y = np.array([int(i > 0.5) for i in y])
savePrediction(y, 'result/prediction.csv')