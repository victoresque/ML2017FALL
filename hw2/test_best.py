import pickle
from sklearn.externals import joblib
from util import *

def test_best(clf, X_test, result_path):
    y_test = clf.predict(X_test).reshape((-1, 1))
    save_prediction(y_test, result_path)