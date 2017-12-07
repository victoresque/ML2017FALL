import sys
from keras.models import Sequential
from keras.layers import Dense, GRU
from preproc import *

print('Loading and preprocessing data...')
x_train, y_train = preprocessTrainingData(sys.argv[1], sys.argv[2])

print('Training...')
model = Sequential()
model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, input_shape=(maxlen, 256)))
model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(512, activation='selu'))
model.add(Dense(512, activation='selu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('rnn.h5')