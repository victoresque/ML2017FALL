import numpy as np

image = np.load('data/image.npy').astype(np.float32)
image = np.reshape(image, (-1, 28, 28, 1))
image = image / 255

from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Dense, Reshape
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

image = np.reshape(image, (-1, 28*28))
input = Input((28*28,))
x = Dense(128, activation='selu')(input)
x = Dense(64, activation='selu')(x)
encoded = Dense(32, activation='selu')(x)
x = Dense(64, activation='selu')(encoded)
x = Dense(128, activation='selu')(x)
output = Dense(28*28, activation='selu')(x)

model = Model(input, output)
encoder = Model(input, encoded)

model.summary()
model.compile(optimizer=Adam(), loss='mse')

callbacks = [ModelCheckpoint('{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5', period=5)]
model.fit(image, image, batch_size=128, epochs=50,
          verbose=1, callbacks=callbacks, validation_split=0.1, shuffle=True)


