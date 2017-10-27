import numpy as np
import pickle
from util import *
from test import *

X_train, y_train = load_train_data('data/X_train.csv', 'data/y_train.csv')
X_test = load_test_data('data/X_test.csv')

do_scale = True
if do_scale:
    X_train, X_test = StandardScale(X_train, X_test)

X_train = to_homogeneous(X_train)
X_test = to_homogeneous(X_test)

def train_logistic():
    targets = y_train.flatten()
    epochs = 100000
    eta0 = 1e-6
    w = np.zeros(X_train.shape[1])
    for i in range(epochs):
        if i % 1000 == 0: print('Epoch', i)
        predictions = sigmoid(np.dot(X_train, w))
        error = targets - predictions
        gradient = np.dot(X_train.T, error)
        w += eta0 * gradient
    return w

w = train_logistic()
pickle.dump(w, open('model/logistic.pkl', 'wb'))

w = pickle.load(open('model/logistic.pkl', 'rb'))
test_logistic(w, X_test, 'result/prediction.csv')
