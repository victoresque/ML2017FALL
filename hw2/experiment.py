import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('data/X_train.csv').values.astype(np.float64)
y_train = pd.read_csv('data/Y_train.csv').values.astype(np.float64)
X_test = pd.read_csv('data/X_test.csv').values.astype(np.float64)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = AdaBoostClassifier(n_estimators=1000)
clf.fit(X_train, y_train.flatten())
y_predict = clf.predict(X_train)
print(accuracy_score(y_train, y_predict))

y = clf.predict(X_test)


'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
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
x = X_test.T
cov_inv = np.linalg.inv(cov)
w = np.dot((u_1 - u_0), cov_inv)
b = (-0.5) * np.dot(np.dot([u_1], cov_inv), u_1) + (0.5) * np.dot(np.dot([u_0], cov_inv), u_0) + np.log(
    float(N_1) / N_0)
y = np.around(sigmoid(np.dot(w, x) + b)).reshape((-1, 1))
'''

pd.DataFrame([[i+1, int(y[i])] for i in range(y.shape[0])], columns=['id', 'label']) \
        .to_csv('result/result.csv', index=False)