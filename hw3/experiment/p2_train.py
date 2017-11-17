import numpy as np
import sys
sys.path.append('..')
from param import *
from util import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model

X_train, y_train = train_data('../data/X_train.npy', '../data/y_train.npy')
X_train, y_train, X_test, y_test = split_valid(X_train, y_train)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
plot_model(model, to_file='img/dnn_structure.png')

model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(X_test, y_test))

model.save('dnn.h5')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])