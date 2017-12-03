import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
lines, labels = readData('data/training_label.txt')
lines = readData('data/pre_training_nolabel.txt', label=False)[:len(lines)]

print('Preprocessing data...')
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
labels = np.array(labels)
print('  Transforming to word vector...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)
print('  Splitting validation...')
x_valid, y_valid = lines[:len(lines)//v], labels[:len(lines)//v]
x_train, y_train = lines[len(lines)//v:], labels[len(lines)//v:]

print('Training...')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, History, ModelCheckpoint

model = Sequential()
model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,
               input_shape=(maxlen, 256)))
model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

history = History()
chkpoint = ModelCheckpoint('model.{epoch:02d}-{val_acc:.4f}.h5',
                           monitor='val_acc',
                           save_best_only=True,
                           save_weights_only=False)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[history, chkpoint],
          validation_data=[x_valid, y_valid])
model.save('rnn_0.h5')
with open('result/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)