import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

lines = readTestData('data/testing_data.txt')
print('Preprocessing data...')
print('  Converting...')
cmap = loadPreprocessCmap('data/pre_cmap.pkl')
transformByConversionMap(lines, cmap)
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
print('  Transforming to word vectors...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)

print('Testing...')
from keras.models import load_model
model = load_model('model.47-0.8408-0.8387.h5')
x_test = lines
y = model.predict(x_test, verbose=True).flatten()
y = np.array([int(i > 0.5) for i in y])
savePrediction(y, 'result/prediction.csv')
