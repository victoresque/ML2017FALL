import numpy as np

image = np.load('data/image.npy')[-100:].astype(np.float32)
image = np.reshape(image, (-1, 28, 28, 1))
image = image / 255

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
model = load_model('09_0.0188_0.0186.h5')
image = np.reshape(image, (-1, 28*28))
recons = model.predict(image, batch_size=64, verbose=1)
image = np.reshape(image, (-1, 28, 28))
recons = np.reshape(recons, (-1, 28, 28))

import cv2
for i in range(100):
    cv2.imshow('Original', image[i])
    cv2.imshow('Reconstructed', recons[i])
    cv2.waitKey()