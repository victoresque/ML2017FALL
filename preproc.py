import pandas as pd
import numpy as np
from param import *
from util import *

data = pd.read_csv('data/train.csv', encoding='big5')\
         .drop(['日期', '測站', '測項'], axis=1)\
         .replace('NR', '0').values

data_flat = np.zeros((n_categories, 0))
for i in range(months_per_year * n_days):
    data_flat = np.append(data_flat, data[i*n_categories:(i+1)*n_categories, :], axis=1)

train_X = np.zeros((0, feature_len))
train_y = np.zeros((0, 1))

for month in range(months_per_year):
    month_offset = month * n_days * hours_per_day
    next_month_offset = (month + 1) * n_days * hours_per_day
    for day in range(n_days):
        day_offset = month_offset + day * hours_per_day
        for time in range(hours_per_day):
            time_offset = day_offset + time
            pm25_col = time_offset + feature_len
            if pm25_col >= next_month_offset:
                break
            x = data_flat[:, time_offset:time_offset + feature_len]
            y = np.array([[data_flat[pm25_category_id][pm25_col]]])

            train_X = np.append(train_X, x, axis=0)
            train_y = np.append(train_y, y, axis=0)

train_X = to_homogeneous(flatten(extract(train_X.astype(np.float))))
pd.DataFrame(train_X).to_csv('preproc/pre_X.csv', index=False, header=False)
pd.DataFrame(train_y).to_csv('preproc/pre_y.csv', index=False, header=False)

