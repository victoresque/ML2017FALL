import numpy as np
from util import *

# from keras.models import Sequential
# from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

X_train, y_train = load_train_data('data/X_train_logistic.csv', 'data/y_train.csv')
data_train = np.append(X_train, y_train, axis=1)
np.random.shuffle(data_train)

v = 16
n = X_train.shape[0]
X_train = data_train[n//v:n, :-1]
y_train = data_train[n//v:n, -1:]
X_valid = data_train[:n//v, :-1]
y_valid = data_train[:n//v, -1:]

scaler = StandardScaler()
scaler.fit_transform(X_train)

clf = RandomForestClassifier(n_estimators=10000, max_depth=16, n_jobs=-1)
clf.fit(X_train, y_train.flatten())
y_predict = clf.predict(X_train)
print(accuracy_score(y_train, y_predict))
y_predict = clf.predict(X_valid)
print(accuracy_score(y_valid, y_predict))

joblib.dump(clf, 'model/RF.pkl')

del clf
clf = joblib.load('model/RF.pkl')

X_test = load_test_data('data/X_test_logistic.csv')
y_test = clf.predict(X_test).reshape((-1, 1))

save_prediction(y_test, 'result/prediction.csv')