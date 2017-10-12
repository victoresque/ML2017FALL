import pandas as pd
import numpy as np
import os
from param import *
from util import *

test_data = pd.read_csv(os.path.join(data_path, 'test.csv'), header=None).replace('NR', '0').as_matrix()
n_rows = test_data.shape[0]//n_categories
test_X = np.zeros((0, x_len))
for i in range(n_rows):
    td = np.zeros((1, 0))
    for cat in categories:
        td = np.append(td, test_data[i*n_categories + cat, -feature_len:])
    test_X = np.append(test_X, [td.flatten().astype(np.float)], axis=0)

test_X = transform(test_X)
test_y = np.dot(w, test_X.T)

pd.DataFrame([['id_'+str(i), test_y[0][i]] for i in range(n_rows)], columns=['id', 'value'])\
  .to_csv(os.path.join(result_path, 'result.csv'), index=False)