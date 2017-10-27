import sys
import pickle
from util import *
from test import *

[method,
 raw_data_path,
 test_data_path,
 train_feature_path,
 train_label_path,
 test_feature_path,
 result_path] = sys.argv[1:8]

X_train, y_train = load_train_data(train_feature_path, train_label_path)
X_test = load_test_data(test_feature_path)

if method == 'generative':
    X_train, X_test = StandardScale(X_train, X_test)
    u_0, u_1, cov, N_0, N_1 = pickle.load(open('model/generative.pkl', 'rb'))
    test_generative(u_0, u_1, cov, N_0, N_1, X_test, result_path)
elif method == 'logistic':
    X_train, X_test = StandardScale(X_train, X_test)
    w = pickle.load(open('model/logistic.pkl', 'rb'))
    test_logistic(w, X_test, result_path)
elif method == 'best':
    clf = joblib.load('model/AdaBoost.pkl')
    test_best(clf, X_test, result_path)
