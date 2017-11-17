import numpy as np
import pandas as pd
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from param import *
from util import *

from keras.utils import plot_model

train_data_path = sys.argv[1]

raw = pd.read_csv(train_data_path)
X_train = raw['feature'].values
X_train = np.array([[int(i) for i in x.split()] for x in X_train]).astype(np.float32)
y_train = raw['label'].values
X_train, y_train = train_data(X_train, y_train)

gen = ImageDataGenerator(
    width_shift_range=0.15,
    height_shift_range=0.1,
    zoom_range=0.8,
    horizontal_flip=True)

gen.fit(X_train)
print(X_train.shape)

if valid:
    X_train, y_train, X_test, y_test = split_valid(X_train, y_train)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png')

if valid:
    model.fit_generator(gen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch = batch_size*4, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
else:
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

model.save('cnn.h5')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])