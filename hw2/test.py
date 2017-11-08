import pickle
from util import *

def test_generative(u_0, u_1, cov, N_0, N_1, X_test, result_path):
    x = X_test.T
    cov_inv = np.linalg.inv(cov)
    w = np.dot((u_1 - u_0), cov_inv)
    b = (-0.5) * np.dot(np.dot([u_1], cov_inv), u_1) + (0.5) * np.dot(np.dot([u_0], cov_inv), u_0) + np.log(
        float(N_1) / N_0)
    y_test = np.around(sigmoid(np.dot(w, x) + b)).reshape((-1, 1))
    save_prediction(y_test, result_path)

def test_logistic(w, X_test, result_path):
    X_test = to_homogeneous(X_test)
    y_test = (sigmoid(np.dot(X_test, w.reshape((-1, 1)))) > 0.5).astype(int)
    save_prediction(y_test, result_path)