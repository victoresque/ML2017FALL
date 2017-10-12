import pandas as pd
import numpy as np
import os
from param import *
from keras.models import load_model

test_data = pd.read_csv('data/test.csv', header=None).replace('NR', '0').values
n_rows = test_data.shape[0]//n_categories
test_X = np.zeros((0, x_len))
for i in range(n_rows):
    td = np.zeros((1, 0))
    for cat in categories:
        td = np.append(td, test_data[i*n_categories + cat, -feature_len:])
    test_X = np.append(test_X, [td.flatten().astype(np.float)], axis=0)

test_y = load_model('model/model.h5').predict(test_X)
pd.DataFrame([['id_'+str(i), test_y[i][0]] for i in range(n_rows)], columns=['id', 'value']) \
          .to_csv('result/result.csv', index=False)