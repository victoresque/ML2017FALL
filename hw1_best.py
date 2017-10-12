import pandas as pd
import numpy as np
from param_best import *
from util_best import *
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

test_X = transform(load_test_data(input_path))
w = load_w('result/w_best.csv')
test_y = np.dot(w, test_X.T)

pd.DataFrame([['id_'+str(i), test_y[0][i]] for i in range(test_X.shape[0])],
             columns=['id', 'value']).to_csv(output_path, index=False)