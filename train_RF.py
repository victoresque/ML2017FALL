import pandas as pd
import numpy as np
from param import *
from util import *
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RANSACRegressor, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.externals import joblib
rmse_cnt = [0, 0, 0, 0, 0, 0]
if not local_valid:
    n_valid = 1

t_rmse = []
for i in range(n_valid):
    v_rmse = []
    train_X, train_y, valid_X, valid_y = load_train_data()

    print(train_X.shape)
    # clf = SVR(kernel='poly', C=1e3, degree=2, verbose=True)
    # clf = RandomForestRegressor(n_estimators=8000, n_jobs=-1, max_depth=4)
    # clf = GradientBoostingRegressor(n_estimators=5000, max_depth=4, min_samples_split=2000)
    clf = Ridge(alpha=1)
    clf.fit(train_X, train_y.flatten())

    test_y = clf.predict(train_X)
    t_rmse.append(rmse(test_y, train_y.T))
    print(rmse(test_y, train_y.T))

    if local_valid:
        for i in range(vn):
            test_X = valid_X[i]
            test_y = clf.predict(test_X)
            v_rmse.append(rmse(test_y, valid_y[i].T))

            rmse_val = rmse(test_y, valid_y[i].T)
            if rmse_val > 7:
                print('++++++++++')
                rmse_cnt[5] += 1
            elif rmse_val > 6.8:
                print('++++++++')
                rmse_cnt[4] += 1
            elif rmse_val > 6.6:
                print('++++++')
                rmse_cnt[3] += 1
            elif rmse_val > 6.4:
                print('++++')
                rmse_cnt[2] += 1
            elif rmse_val > 6.2:
                print('++')
                rmse_cnt[1] += 1
            else:
                print('')
                rmse_cnt[0] += 1
        print(v_rmse)
        print(['{0:.1f}%'.format(e * 100 / np.sum(rmse_cnt)) for e in rmse_cnt])
    else:
        test_X = to_homogeneous(flatten(extract(load_test_data())))
        test_y = clf.predict(test_X)
        # test_y = clamp(test_y, test_X[:, feature_len])
        pd.DataFrame([['id_'+str(i), test_y[i]] for i in range(test_X.shape[0])],
                     columns=['id', 'value']).to_csv('result/result.csv', index=False)

joblib.dump(clf, 'model/model_RF.pkl')

if local_valid:
    print('')
    print('----------------------------------------')
    print('   Training   mean =', np.mean(t_rmse))
    print('   Training   std  =', np.std(t_rmse))
    print('   Validation mean =', np.mean(v_rmse))
    print('   Validation std  =', np.std(v_rmse))
    print('   Difference      =', np.mean(v_rmse)-np.mean(t_rmse))
    print('----------------------------------------')