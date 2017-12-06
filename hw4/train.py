import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
lines, labels = readData('data/training_label.txt')
lines = readData('data/pre_corpus.txt', label=False)[:len(lines)]
shuffleData(lines, labels)

print('Preprocessing data...')
print('  Converting...')
cmap = loadPreprocessCmap('data/pre_cmap.pkl')
transformByConversionMap(lines, cmap)
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
labels = np.array(labels)
print('  Transforming to word vector...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)
print('  Splitting validation...')
v = 20000
x_valid, y_valid = lines[:v], labels[:v]
x_train, y_train = lines[v:], labels[v:]

print('Training...')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, History, ModelCheckpoint

model = Sequential()
model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,
               input_shape=(maxlen, 256)))
model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(512, activation='selu'))
model.add(Dense(512, activation='selu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()
history = History()
chkpoint = ModelCheckpoint('model.{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5',
                           monitor='val_acc',
                           save_best_only=False,
                           save_weights_only=False)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[history, chkpoint],
          validation_data=[x_valid, y_valid])