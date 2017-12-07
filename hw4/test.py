import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from preproc import *

x_test = loadTestingData('data/testing_data.txt')

model_cnt = 8
ylist = []
for i in range(model_cnt):
    model = load_model('model/0.83955/m'+str(i+1)+'.h5')
    ylist.append(model.predict(x_test, batch_size=512, verbose=True).flatten())

y = np.zeros(len(ylist[0]))
for i, yi in enumerate(ylist):
    y = y + yi
y = y / model_cnt
y = np.array([int(i > 0.5) for i in y])
savePrediction(y, 'result/prediction.csv')
