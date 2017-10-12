import pandas as pd
import numpy as np
from param import *
from util import *

test_X = to_homogeneous(flatten(extract(load_test_data())))
test_y = np.zeros((1, test_X.shape[0]))

for i in range(test_X.shape[0]):
    x = test_X[i][feature_len]
    x1 = test_X[i][feature_len - 1]
    x2 = test_X[i][feature_len - 2]
    r = 1.0
    if (x - x1) * (x1 - x2) <= 0:
        r = 0.2
    test_y[0][i] = x + ((x - x1) * 0.6 + (x1 - x2) * 0.4) * r
    if test_y[0][i] < 0:
        test_y[0][i] = 0

pd.DataFrame([['id_'+str(i), test_y[0][i]] for i in range(test_X.shape[0])],
             columns=['id', 'value']).to_csv('result/result.csv', index=False)