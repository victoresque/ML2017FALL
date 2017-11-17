import numpy as np
import keras
import os
from param import *

from PIL import Image

def train_data(X, y):
    X_train = X / 255
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    y_train = y
    y_train = keras.utils.to_categorical(y_train, num_classes)
    return X_train, y_train

def split_valid(X_train, y_train):
    X_test = X_train[:X_train.shape[0] // v, :, :, :]
    y_test = y_train[:y_train.shape[0] // v, :]
    X_train = X_train[X_train.shape[0] // v:, :, :, :]
    y_train = y_train[y_train.shape[0] // v:, :]
    return X_train, y_train, X_test, y_test

def to_categorical(y, n_class):
    return keras.utils.to_categorical(y, n_class)