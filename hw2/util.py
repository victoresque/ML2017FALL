import numpy as np
import pandas as pd

def load_train_data(X_path, y_path):
    X_train = pd.read_csv(X_path).drop('fnlwgt', axis=1).values.astype(np.float64)
    y_train = pd.read_csv(y_path).values.astype(np.float64)
    return X_train, y_train

def load_test_data(X_path):
    X_test = pd.read_csv(X_path).drop('fnlwgt', axis=1).values.astype(np.float64)
    return X_test

def save_prediction(y, prediction_path):
    pd.DataFrame([[i+1, int(y[i][0])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv(prediction_path, index=False)

def to_homogeneous(X):
    return np.insert(X, 0, [1], axis=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def StandardScale(X_train, X_test):
    mean = np.mean(X_train, axis=0).reshape((1, -1))
    std = np.std(X_train, axis=0).reshape((1, -1))
    return ((X_train - mean) / std), ((X_test - mean) / std)
