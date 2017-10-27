import numpy as np
from util import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

X_train, y_train = load_train_data('data/X_train.csv', 'data/y_train.csv')

data_train = np.append(X_train, y_train, axis=1)

valid = False
if valid:
    np.random.shuffle(data_train)
    v = 3
    n = X_train.shape[0]
    X_train = data_train[n//v:n, :-1]
    y_train = data_train[n//v:n, -1:]
    X_valid = data_train[:n//v, :-1]
    y_valid = data_train[:n//v, -1:]

clf = AdaBoostClassifier(n_estimators=1000)
clf.fit(X_train, y_train.flatten())
y_predict = clf.predict(X_train)
print(accuracy_score(y_train, y_predict))

if valid:
    y_predict = clf.predict(X_valid)
    print(accuracy_score(y_valid, y_predict))

joblib.dump(clf, 'model/AdaBoost.pkl')

clf = joblib.load('model/AdaBoost.pkl')
X_test = load_test_data('data/X_test.csv')
y_test = clf.predict(X_test).reshape((-1, 1))
save_prediction(y_test, 'result/prediction.csv')