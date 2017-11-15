import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from param import *
from util import *

X_train, y_train = train_data()
# = facial_extract(X_train)

gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

gen.fit(X_train)
print(X_train.shape)

if valid:
    X_train, y_train, X_test, y_test = split_valid(X_train, y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
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
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

if valid:
    model.fit_generator(gen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch = len(X_train)//32, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
else:
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

model.save('model/cnn.h5')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])