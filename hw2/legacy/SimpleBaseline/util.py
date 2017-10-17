import numpy as np
import pandas as pd

def load_train_data(X_path, y_path):
    X_train = pd.read_csv(X_path).values.astype(np.float64)
    y_train = pd.read_csv(y_path).values.astype(np.float64)
    return X_train, y_train

def load_test_data(X_path):
    X_test = pd.read_csv(X_path).values.astype(np.float64)
    return X_test

def save_prediction(y, prediction_path):
    pd.DataFrame([[i+1, int(y[i][0])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv(prediction_path, index=False)

def MultivariateGaussian(miu, cov, x):
    D = miu.shape[0]
    c1 = 1 / np.power(2 * np.pi, D/2)
    c2 = 1 / np.sqrt(np.abs(np.linalg.det(cov)))
    p = np.exp(-0.5 * np.matmul((x - miu).T, np.matmul(np.linalg.inv(cov), (x - miu))))[0][0]
    return c1 * c2 * p

def to_homogeneous(X):
    return np.insert(X, 0, [1], axis=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


