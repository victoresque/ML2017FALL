import pandas as pd
import numpy as np
from param import *
from util import *
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.externals import joblib

test_X = to_homogeneous(flatten(extract(load_test_data())))

clf = joblib.load('model/model_RF.pkl')
test_y = clf.predict(test_X)
test_y = clamp(test_y, test_X[:, feature_len])

pd.DataFrame([['id_'+str(i), test_y[i]] for i in range(test_X.shape[0])],
             columns=['id', 'value']).to_csv('result/result.csv', index=False)