import numpy as np
from util import *
from sklearn.linear_model import LogisticRegression

def train_logistic():
    X_train, y_train = load_train_data('data/X_train_logistic.csv', 'data/y_train.csv')
    X_train = to_homogeneous(X_train)
    epochs = 1000
    eta0 = 1e-12
    w = np.zeros((X_train.shape[1], 1))
    for i in range(epochs):
        if i % 100 == 0:
            print('Epoch', i)
        eta = eta0 / np.sqrt(i + 1)
        y_f = y_train - sigmoid(np.matmul(X_train, w))
        gradient = np.matmul(y_f.T, X_train).T
        w += eta * gradient

    return w

def test_logistic(w):
    X_test = load_test_data('data/X_test.csv')
    X_test = to_homogeneous(X_test)
    y_test = np.zeros((X_test.shape[0], 1))
    for i in range(X_test.shape[0]):
        x = X_test[i]
        label_infer = sigmoid(np.dot(w.flatten(), x)) > 0.5
        y_test[i][0] = label_infer

    save_prediction(y_test, 'result/prediction.csv')

def test_train_set(w):
    X_train, y_train = load_train_data('data/X_train_logistic.csv', 'data/y_train.csv')
    X_train = to_homogeneous(X_train)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(X_train.shape[0]):
        x = X_train[i]
        label_infer = sigmoid(np.dot(w.flatten(), x)) > 0.5
        label_truth = y_train[i][0]

        TP += (label_infer == True) and (label_truth == True)
        TN += (label_infer == False) and (label_truth == False)
        FP += (label_infer == True) and (label_truth == False)
        FN += (label_infer == False) and (label_truth == True)
        print('Accuracy =', (TP + TN) / (TP + TN + FP + FN))

w = train_logistic()

test_train_set(w)



'''
    X_train, y_train = load_train_data('data/X_train_logistic.csv', 'data/y_train.csv')
    data_train = np.append(X_train, y_train, axis=1)
    np.random.shuffle(data_train)
    v = 8
    n = X_train.shape[0]
    X_train = data_train[n // v:n, :-1]
    y_train = data_train[n // v:n, -1:]
    X_valid = data_train[:n // v, :-1]
    y_valid = data_train[:n // v, -1:]
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.score(X_valid, y_valid))
'''