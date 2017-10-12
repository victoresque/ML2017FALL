import pandas as pd
import numpy as np
from param import *
from util import *

rmse_cnt = [0, 0, 0, 0, 0, 0]
if not local_valid:
    n_valid = 1

t_rmse = []
for i in range(n_valid):
    print('.', end='', flush=True)

    v_rmse = []
    train_X, train_y, valid_X, valid_y = load_train_data()
    w = np.zeros((1, train_X.shape[1]))

    if use_gradient_descent:
        # Gradient descent using Adagrad
        g_norm_sum = 0.0
        for i in range(n_epoch):
            eta, g = eta0/np.sqrt(i+1), gradient(w, train_X, train_y, Lambda)
            g_norm_sum += np.square(np.linalg.norm(g))
            w = w - eta * g / np.sqrt(g_norm_sum/(i+1))
            print('Epoch', str(i)+': rmse =', rmse(np.dot(w, train_X.T), train_y.T))
    else:
        # using closed-form solution
        reg_I = Lambda * train_X.shape[0] * np.identity(train_X.shape[1])
        w = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T, train_X) + reg_I),
                          train_X.T), train_y).T
        t_rmse.append(rmse(np.dot(w, train_X.T), train_y.T))
    if local_valid:
        for i in range(vn):
            test_X = valid_X[i]
            test_y = np.dot(w, test_X.T)
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
        print(['{0:.1f}%'.format(e * 100 / np.sum(rmse_cnt)) for e in rmse_cnt])
    else:
        test_X = to_homogeneous(flatten(extract(load_test_data())))
        test_y = np.dot(w, test_X.T)
        pd.DataFrame([['id_'+str(i), test_y[0][i]] for i in range(test_X.shape[0])],
                     columns=['id', 'value']).to_csv('result/result.csv', index=False)

    pd.DataFrame(w.T).to_csv('result/w.csv', header=None, index=False)

if local_valid:
    print('')
    print('----------------------------------------')
    print('   Training   mean =', np.mean(t_rmse))
    print('   Training   std  =', np.std(t_rmse))
    print('   Validation mean =', np.mean(v_rmse))
    print('   Validation std  =', np.std(v_rmse))
    print('   Difference      =', np.mean(v_rmse)-np.mean(t_rmse))
    print('----------------------------------------')