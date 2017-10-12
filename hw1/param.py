# Constants
months_per_year = 12
hours_per_day = 24

# Data-specific constants
n_days = 20
n_categories = 18
pm25_category_id = 9

# Adjustable
feature_len = 9
categories = [2, 3, 7, 8, 9]
cat_order  = [1, 1, 1, 1, 1]

local_valid = False
n_valid = 10
Lambda = 10

valid_fold = 20
use_gradient_descent = True
eta0 = 10
n_epoch = 100000

# Derived constants
x_len = feature_len * len(categories)