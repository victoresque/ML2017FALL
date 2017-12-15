import pandas as pd 
import numpy as np 

pred = pd.read_csv('prediction.csv').values[:,1]
id = np.arange(5000)
np.random.shuffle(id)

for i in range(200):
    pred[id[i]] = np.random.randint(6)

with open('prediction2.csv', 'w') as f:
    f.write('id,ans\n')
    for i, a in enumerate(pred):
        f.write(str(i+1)+','+str(a)+'\n')