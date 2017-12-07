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
ylist = []
for i in range(8):
    model = load_model('ensembel_best/new/m'+str(i+1)+'.h5')
    ylist.append(model.predict(x_test, batch_size=bs, verbose=True).flatten())

with open('result/y.pkl', 'wb') as f:
    pickle.dump(ylist, f)

with open('result/y.pkl', 'rb') as f:
    ylist = pickle.load(f)

y = np.zeros(len(ylist[0]))
for i, yi in enumerate(ylist):
    y = y + yi
y = y / 8
y = np.array([int(i > 0.5) for i in y])
savePrediction(y, 'result/prediction.csv')