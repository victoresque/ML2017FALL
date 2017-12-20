import pandas as pd
from util import *

p1 = pd.read_csv('result/p1.csv').values[:,1]
p2 = pd.read_csv('result/p2.csv').values[:,1]

p = (p1 + p2) / 2
print(p)

savePrediction(p, 'result/p_ensemble.csv')