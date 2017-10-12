import pandas as pd
import numpy as np
import os
from param_best import *
from util import *

# Splitting training X and y
t_rmse = []
v_rmse = []
for i in range(n_valid):
    print('.', end='', flush=True)
    train_data = pd.read_csv(os.path.join(preproc_path, 'preproc.csv'), header=None).as_matrix()
    if local_valid:
        np.random.shuffle(train_data)
    n_rows, n_cols = train_data.shape[0], train_data.shape[1]
    train_X = transform(np.array(train_data[0:n_rows, 0:n_cols-1]))
    train_y = np.array([train_data[0:n_rows, n_cols-1]])
    if i == 0:
        print(train_X.shape)

    if local_valid:
        valid_X = train_X[0:(n_rows//valid_fold),:]
        valid_y = train_y[:,0:(n_rows//valid_fold)]
        train_X = train_X[(n_rows//valid_fold):n_rows,:]
        train_y = train_y[:,(n_rows//valid_fold):n_rows]

    w = np.zeros((1, train_X.shape[1]))
    if use_gradient_descent:
        # Gradient descent using Adagrad
        g_norm_sum = 0.0
        for i in range(n_epoch):
            eta, g = eta0/np.sqrt(i+1), gradient(w, train_X, train_y, Lambda)
            g_norm_sum += np.square(np.linalg.norm(g))
            w = w - eta*g/np.sqrt(g_norm_sum/(i+1))
            print('Epoch', str(i)+': rmse =', rmse(np.dot(w, train_X.T), train_y))
    else:
        w = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T, train_X) + Lambda * np.identity(train_X.shape[1])),
                          train_X.T), train_y.T).T
        #print('Training   rmse =', rmse(np.dot(w, train_X.T), train_y))
        t_rmse.append(rmse(np.dot(w, train_X.T), train_y))

    test_data = load_test_data('data/test.csv')

    if local_valid:
        test_X = valid_X
        test_y = np.dot(w, test_X.T)
        #print('Validation rmse =', rmse(test_y, valid_y))
        v_rmse.append(rmse(test_y, valid_y))
    else:
        test_X = transform(test_X)
        test_y = np.dot(w, test_X.T)
        pd.DataFrame([['id_' + str(i), test_y[0][i]] for i in range(n_rows)], columns=['id', 'value']) \
          .to_csv(os.path.join(result_path, 'result.csv'), index=False)

    pd.DataFrame(w.T)\
      .to_csv(os.path.join(result_path, 'w.csv'), header=None, index=False)

    if not local_valid:
        break

if local_valid:
    print('')
    print('----------------------------------------')
    print('   Training   mean =', np.mean(t_rmse))
    print('   Training   std  =', np.std(t_rmse))
    print('   Validation mean =', np.mean(v_rmse))
    print('   Validation std  =', np.std(v_rmse))
    print('   Difference      =', np.mean(v_rmse)-np.mean(t_rmse))
    print('----------------------------------------')