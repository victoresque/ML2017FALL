import pandas as pd
import numpy as np
import os
from param import *

test_path = os.path.join(data_path, 'test.csv')
test_data = pd.read_csv(test_path, encoding='big5')