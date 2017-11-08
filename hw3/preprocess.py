import numpy as np
import pandas as pd

raw = pd.read_csv('data/train.csv')
X_train = raw['feature'].values
X_train = np.array([[int(i) for i in x.split()] for x in X_train]).astype(np.float32)
y_train = raw['label'].values

raw = pd.read_csv('data/test.csv')
X_test = raw['feature']
X_test = np.array([[int(i) for i in x.split()] for x in X_test]).astype(np.float32)

np.save('data/X_train', X_train)
np.save('data/y_train', y_train)
np.save('data/X_test', X_test)