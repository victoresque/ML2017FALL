import numpy as np
from util import *

def train_generative():
    X_train, y_train = load_train_data('data/X_train_generative.csv', 'data/y_train.csv')
    X_train_0 = X_train[y_train[:, 0] == 0, :]
    X_train_1 = X_train[y_train[:, 0] == 1, :]

    cov = np.cov(X_train.T)
    u_0 = np.mean(X_train_0, axis=0, keepdims=True).T
    u_1 = np.mean(X_train_1, axis=0, keepdims=True).T
    p_0 = X_train_0.shape[0] / X_train.shape[0]
    p_1 = X_train_1.shape[0] / X_train.shape[0]

    return u_0, u_1, p_0, p_1, cov

def test_generative(u_0, u_1, p_0, p_1, cov):
    X_test = load_test_data('data/X_test.csv')
    y_test = np.zeros((X_test.shape[0], 1))
    for i in range(X_test.shape[0]):
        x = X_test[i:i + 1, :]
        p_x_0 = MultivariateGaussian(u_0, cov, x.T)
        p_x_1 = MultivariateGaussian(u_1, cov, x.T)

        p_0_x = p_x_0 * p_0 / (p_x_0 * p_0 + p_x_1 * p_1)

        label_infer = p_0_x < 0.5
        y_test[i][0] = label_infer

    save_prediction(y_test, 'result/prediction.csv')

def test_train_set(u_0, u_1, p_0, p_1, cov):
    X_train, y_train = load_train_data('data/X_train_generative.csv', 'data/y_train.csv')
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(X_train.shape[0]):
        x = X_train[i:i+1, :]
        p_x_0 = MultivariateGaussian(u_0, cov, x.T)
        p_x_1 = MultivariateGaussian(u_1, cov, x.T)

        p_0_x = p_x_0 * p_0 / (p_x_0 * p_0 + p_x_1 * p_1)
        label_infer = p_0_x < 0.5
        label_truth = y_train[i][0]

        TP += (label_infer == True) and (label_truth == True)
        TN += (label_infer == False) and (label_truth == False)
        FP += (label_infer == True) and (label_truth == False)
        FN += (label_infer == False) and (label_truth == True)
        print('Accuracy =', (TP + TN) / (TP + TN + FP + FN))


u_0, u_1, p_0, p_1, cov = train_generative()

test_train_set(u_0, u_1, p_0, p_1, cov)
# test_generative(u_0, u_1, p_0, p_1, cov)
