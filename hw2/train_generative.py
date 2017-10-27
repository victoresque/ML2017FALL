import numpy as np
import pickle
from util import *
from test import *

X_train, y_train = load_train_data('data/X_train.csv', 'data/y_train.csv')
X_test = load_test_data('data/X_test.csv')

do_scale = True
if do_scale:
    X_train, X_test = StandardScale(X_train, X_test)

def train_generative():
    X_train_0 = X_train[y_train[:, 0] == 0, :]
    X_train_1 = X_train[y_train[:, 0] == 1, :]

    N_0 = X_train_0.shape[0]
    N_1 = X_train_1.shape[0]

    cov_0 = np.cov(X_train_0.T)
    cov_1 = np.cov(X_train_1.T)
    cov = cov_0 * (N_0 / (N_0 + N_1)) + cov_1 * (N_1 / (N_0 + N_1))
    u_0 = np.mean(X_train_0, axis=0, keepdims=True).flatten()
    u_1 = np.mean(X_train_1, axis=0, keepdims=True).flatten()

    return u_0, u_1, cov, N_0, N_1

u_0, u_1, cov, N_0, N_1 = train_generative()
params = u_0, u_1, cov, N_0, N_1
pickle.dump(params, open('model/generative.pkl', 'wb'))

u_0, u_1, cov, N_0, N_1 = pickle.load(open('model/generative.pkl', 'rb'))
test_generative(u_0, u_1, cov, N_0, N_1, X_test, 'result/prediction.csv')
