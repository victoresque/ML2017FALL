import numpy as np
import keras
import os
from param import *

from PIL import Image

def train_data(X, y):
    X_train = np.load(X)/255
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    y_train = np.load(y)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    return X_train, y_train

def split_valid(X_train, y_train):
    X_test = X_train[:X_train.shape[0] // v, :, :, :]
    y_test = y_train[:y_train.shape[0] // v, :]
    X_train = X_train[X_train.shape[0] // v:, :, :, :]
    y_train = y_train[y_train.shape[0] // v:, :]
    return X_train, y_train, X_test, y_test

def facial_extract(X):
    r, c = img_rows, img_cols
    X_ext = []
    for i in range(X.shape[0]):
        x = np.reshape(X[i], (r, c))
        eye = x[r//5:r//2, c//5:4*c//5]
        mouth = x[2*r//3:, c//4:3*c//4]
        width = max(eye.shape[1], mouth.shape[1])
        eye = np.lib.pad(eye, ((0, 0), (0, width - eye.shape[1])), 'constant')
        mouth = np.lib.pad(mouth, ((0, 0), (0, width - mouth.shape[1])), 'constant')
        X_ext.append(np.append(eye, mouth, axis=0))
        X_ext[i] = np.reshape(X_ext[i], (X_ext[i].shape[0], X_ext[i].shape[1], 1))
    return np.array(X_ext)

def to_categorical(y, n_class):
    return keras.utils.to_categorical(y, n_class)