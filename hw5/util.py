import numpy as np
import pandas as pd

def savePrediction(y, path, id_start=1):
    y = np.array(y)
    pd.DataFrame([[i+id_start, y[i]] for i in range(y.shape[0])],
                 columns=['TestDataID', 'Rating']).to_csv(path, index=False)
