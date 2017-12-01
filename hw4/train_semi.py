import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
lines = []
with open('data/pre_training_nolabel.txt', 'r', encoding='utf_8') as f:
    for line in f:
        lines.append(line.split())
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
print('  Transforming to word vectors...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)

print('Generating pseudo label...')
from keras.models import load_model

model = load_model('rnn_0.h5')
x_test = lines
y = model.predict(x_test, verbose=True).flatten()

lines = []
labels = []
for i, yi in enumerate(y):
    if yi >= 0.9:
        lines.append(x_test[i])
        labels.append(1)
    elif yi <= 0.1:
        lines.append(x_test[i])
        labels.append(0)

print('  Splitting validation...')
x_valid, y_valid = lines[:len(lines)//v], labels[:len(lines)//v]
x_train, y_train = lines[len(lines)//v:], labels[len(lines)//v:]

print('Training...')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

model = Sequential()
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,
               input_shape=(maxlen, 128)))
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_valid, y_valid])
model.save('rnn_0_semi.h5')