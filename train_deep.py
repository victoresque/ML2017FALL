import numpy as np
import pandas as pd
from param import *
from util import *
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model

train_data = pd.read_csv('preproc/train.csv', header=None).values
if local_valid:
    np.random.shuffle(train_data)
n_rows = train_data.shape[0]
n_cols = train_data.shape[1]
train_X = train_data[:, 0:n_cols-1]
train_y = np.array([train_data[:, n_cols-1]]).T

if local_valid:
    v = n_rows//valid_fold//vn
    valid_X = [train_X[v*i:v*(i+1),:] for i in range(vn)]
    valid_y = [train_y[v*i:v*(i+1),:] for i in range(vn)]
    train_X = train_X[v*vn:n_rows,:]
    train_y = train_y[v*vn:n_rows,:]

model = Sequential()
xdim = train_X.shape[1]
model.add(Dense(6, input_dim=xdim,
                activation='relu',
                kernel_regularizer=l2(1e-4)))
model.add(Dense(1, activation='relu',
                kernel_regularizer=l2(1e-2)))

model.compile(loss='mse', optimizer='adam')
model.fit(train_X, train_y, epochs=100, verbose=1)

score_t = np.sqrt(model.evaluate(train_X, train_y, verbose=0))
print('', flush=True)
print(score_t, '\n')
if local_valid:
    for i in range(vn):
        score_v = np.sqrt(model.evaluate(valid_X[i], valid_y[i], verbose=0))
        print('', flush=True)
        print(score_v)

test_data = pd.read_csv('data/test.csv', header=None).replace('NR', '0').values
n_rows = test_data.shape[0]//n_categories
test_X = np.zeros((0, x_len))
for i in range(n_rows):
    td = np.zeros((1, 0))
    for cat in categories:
        td = np.append(td, test_data[i*n_categories + cat, -feature_len:])
    test_X = np.append(test_X, [td.flatten().astype(np.float)], axis=0)


model.save('model/model.h5')
del model
model = load_model('model/model.h5')

test_y = model.predict(test_X)
pd.DataFrame([['id_' + str(i), test_y[i][0]] for i in range(n_rows)], columns=['id', 'value']) \
          .to_csv(os.path.join(result_path, 'result.csv'), index=False)