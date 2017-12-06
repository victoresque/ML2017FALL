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
x_test = lines
bs = 256
model = load_model('ensembel_best/new/m1.h5')
y1 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m2.h5')
y2 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m3.h5')
y3 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m4.h5')
y4 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m5.h5')
y5 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m6.h5')
y6 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m7.h5')
y7 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m8.h5')
y8 = model.predict(x_test, batch_size=bs, verbose=True).flatten()
model = load_model('ensembel_best/new/m9.h5')
y9 = model.predict(x_test, batch_size=bs, verbose=True).flatten()

with open('result/y.pkl', 'wb') as f:
    pickle.dump([y1, y2, y3, y4, y5, y6, y7, y8, y9], f)

with open('result/y.pkl', 'rb') as f:
    [y1, y2, y3, y4, y5, y6, y7, y8, y9] = pickle.load(f)

y = (y1+y2+y3+y4+y5+y6+y7+y8+y9)/9
y = np.array([int(i > 0.5) for i in y])

savePrediction(y, 'result/prediction.csv')