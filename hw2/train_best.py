import numpy as np
from util import *

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

X_train, y_train = load_train_data('data/X_train_logistic.csv', 'data/y_train.csv')

data_train = np.append(X_train, y_train, axis=1)
scaler = StandardScaler()

valid = True

if valid:
    np.random.shuffle(data_train)
    v = 3
    n = X_train.shape[0]
    X_train = data_train[n//v:n, :-1]
    y_train = data_train[n//v:n, -1:]
    X_valid = data_train[:n//v, :-1]
    y_valid = data_train[:n//v, -1:]

clf0 = RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1)
clf22 = AdaBoostClassifier(n_estimators=1000)

clf1 = XGBClassifier(n_estimators=100, n_jobs=-1, max_depth=8)
clf2 = XGBClassifier(n_estimators=200, n_jobs=-1, max_depth=6)
clf3 = XGBClassifier(n_estimators=400, n_jobs=-1, max_depth=4)
clf4 = XGBClassifier(n_estimators=800, n_jobs=-1, max_depth=3)
clf5 = XGBClassifier(n_estimators=1600, n_jobs=-1, max_depth=2)

clf = VotingClassifier(estimators=[
         ('xgb1', clf1), ('xgb2', clf2), ('xgb3', clf3), ('xgb4', clf4), ('xgb5', clf5)], voting='hard')

clf = clf3

X_train = scaler.fit_transform(X_train)
if valid:
    X_valid = scaler.transform(X_valid)
clf.fit(X_train, y_train.flatten())
y_predict = clf.predict(X_train)
print(accuracy_score(y_train, y_predict))

if valid:
    y_predict = clf.predict(X_valid)
    print(accuracy_score(y_valid, y_predict))

joblib.dump(clf, 'model/model.pkl')

del clf
clf = joblib.load('model/model.pkl')

X_test = load_test_data('data/X_train_logistic.csv')
y_test = clf.predict(X_test).reshape((-1, 1))

save_prediction(y_test, 'result/prediction.csv')