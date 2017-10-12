import pandas as pd
import numpy as np
from param import *
from util import *

test_X = to_homogeneous(flatten(extract(load_test_data())))

w = load_w()
test_y = np.dot(w, test_X.T)

pd.DataFrame([['id_'+str(i), test_y[0][i]] for i in range(test_X.shape[0])],
             columns=['id', 'value']).to_csv('result/result.csv', index=False)