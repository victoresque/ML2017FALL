import numpy as np
from util import *
from param import *

print('Loading and preprocessing data...')
lines, labels = readData('data/training_label.txt')
lines = preprocessLines(lines)

dictionary = getDictionary(lines)
print('Dictionary size:', len(dictionary))
lines = transformByDictionary(lines, dictionary)
x_valid, y_valid = lines[:len(lines)//v], labels[:len(lines)//v]
x_train, y_train = lines[len(lines)//v:], labels[len(lines)//v:]


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
print('(train, test) =', len(x_train), len(x_valid))
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_valid = sequence.pad_sequences(x_valid, maxlen=maxlen)
y_train = np.array(y_train)
y_valid = np.array(y_valid)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_valid, y_valid])

model.save('rnn_0.h5')