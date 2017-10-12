# Paths
data_path = 'data'
preproc_path = 'preproc'
result_path = 'result'

# Constants
hours_per_day = 24

# Data-specific constants
n_days = 240
n_categories = 18
pm25_category_id = 9

# Adjustable
feature_len = 9 # need pre-process after changing
categories = [8, 9, 2, 3, 7] # need pre-process after changing
cat_2nd = [0, 1, 4]
cat_all2nd = [0, 1]
lookback_2nd = 5
lookback_all_2nd = 3
cat_3rd = []

local_valid = False
n_valid = 1000
Lambda = 1e3

valid_fold = 20
use_gradient_descent = False
eta0 = 1
n_epoch = 10000

# Derived constants
x_len = feature_len * len(categories)