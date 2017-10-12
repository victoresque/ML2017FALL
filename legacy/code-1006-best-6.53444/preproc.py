import pandas as pd
import numpy as np
import os
from param import *

raw_data = pd.read_csv(os.path.join(data_path, 'train.csv'), encoding='big5')
train_data = raw_data.drop(['日期', '測站', '測項'], axis=1)\
                     .replace('NR', '0').as_matrix().astype(np.float)

train_x, train_y = np.zeros((0, x_len)), np.zeros((0, 1))

for day in range(n_days):
    for time in range(hours_per_day):
        pm25_row = day*n_categories + pm25_category_id + (time+feature_len>=hours_per_day)*n_categories
        pm25_col = (time+feature_len) % hours_per_day
        if pm25_row > train_data.shape[0]:
            break
        x, y = np.zeros((1, x_len)), np.array([[train_data[pm25_row][pm25_col]]])
        for ci in range(len(categories)):
            for i in range(feature_len):
                row = day*n_categories + categories[ci] + (time+i>=hours_per_day)*n_categories
                col = (time+i) % hours_per_day
                x[0][ci*feature_len + i] = train_data[row][col]
        train_x, train_y = np.append(train_x, x, axis=0), np.append(train_y, y, axis=0)

pd.DataFrame(np.append(train_x, train_y, axis=1))\
  .to_csv(os.path.join(preproc_path, 'preproc.csv'), index=False, header=False)

