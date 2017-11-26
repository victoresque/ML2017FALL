import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from util import *
from param import *

lines, labels = readData('data/training_label.txt')
lines, dictionary = getDictionaryAndTransform(lines)

print('Loading data...')
x_test = lines[:len(lines)//v]
y_test = labels[:len(lines)//v]
x_train = lines[len(lines)//v:]
y_train = labels[len(lines)//v:]

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_test, y_test])

model.save('rnn_0.h5')

#lines_nolabel = readData('data/training_nolabel.txt', label=False)