import numpy as np

# Constants
months_per_year = 12
hours_per_day = 24

# Data-specific constants
n_days = 20
n_categories = 18
pm25_category_id = 9

# Adjustable
feature_len = 9
categories = [9, 8, 2, 3, 4, 5, 6, 7, 11, 12]
cat_order  = [1, 1, 1, 1, 1, 1, 1, 1, 1,  1 ]
clamp_thres = 50

local_valid = True
n_valid = 100
Lambda = 1e-3

valid_fold = 3
vn = 8
use_gradient_descent = False
eta0 = 10
n_epoch = 100000

# Derived constants
x_len = np.sum(cat_order)
