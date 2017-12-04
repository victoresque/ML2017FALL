import numpy as np
import pandas as pd
from util import *

p1 = pd.read_csv('result/p1.csv').values[:,1]
p2 = pd.read_csv('result/p2.csv').values[:,1]
p3 = pd.read_csv('result/p3.csv').values[:,1]
p4 = pd.read_csv('result/p4.csv').values[:,1]
p5 = pd.read_csv('result/p5.csv').values[:,1]

p = p1+p2+p3+p4+p5
y = np.array([int(i > 2) for i in p])
savePrediction(y, 'result/prediction.csv')