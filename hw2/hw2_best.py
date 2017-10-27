import sys
import pickle
from util import *
from test_best import *

[method,
 raw_data_path,
 test_data_path,
 train_feature_path,
 train_label_path,
 test_feature_path,
 result_path] = sys.argv[1:8]

X_train, y_train = load_train_data(train_feature_path, train_label_path)
X_test = load_test_data(test_feature_path)

if method == 'best':
    clf = joblib.load('model/AdaBoost.pkl')
    test_best(clf, X_test, result_path)
