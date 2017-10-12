import pandas as pd
import numpy as np
from param import *
from util import *
from sklearn.svm import SVR

t_rmse, v_rmse = [], []
if not local_valid:
    n_valid = 1
for i in range(n_valid):
    train_X, train_y, valid_X, valid_y = load_train_data()
    clf = SVR(C=3e5, gamma=1e-6, verbose=True)
    clf.fit(train_X, train_y.flatten())

    test_y = clf.predict(train_X)
    t_rmse.append(rmse(test_y, train_y.T))

    if local_valid:
        for i in range(vn):
            test_X = valid_X[i]
            test_y = clf.predict(test_X)
            v_rmse.append(rmse(test_y, valid_y[i].T))
            print(rmse(test_y, valid_y[i].T))
    else:
        test_X = to_homogeneous(flatten(extract(load_test_data())))
        test_y = clf.predict(test_X)
        pd.DataFrame([['id_'+str(i), test_y[0][i]] for i in range(test_X.shape[0])],
                     columns=['id', 'value']).to_csv('result/result.csv', index=False)

if local_valid:
    print('')
    print('----------------------------------------')
    print('   Training   mean =', np.mean(t_rmse))
    print('   Training   std  =', np.std(t_rmse))
    print('   Validation mean =', np.mean(v_rmse))
    print('   Validation std  =', np.std(v_rmse))
    print('   Difference      =', np.mean(v_rmse)-np.mean(t_rmse))
    print('----------------------------------------')