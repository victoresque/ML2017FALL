import sys
import numpy as np
import pandas as pd

image_path = sys.argv[1]
test_path = sys.argv[2]
pred_path = sys.argv[3]

def savePrediction(y, path, id_start=0):
    y = np.array(y)
    pd.DataFrame([[i+id_start, y[i]] for i in range(y.shape[0])],
                 columns=['ID', 'Ans']).to_csv(path, index=False)

image = np.load(image_path).astype(np.float32)
image = np.reshape(image, (-1, 28, 28, 1))
image = image / 255

test_data = pd.read_csv(test_path).values[:, 1:]
clust = np.load('data/clust.npy')

numclust = {1, 3, 4, 6, 9, 13, 19}
'''
import cv2
for i, c in enumerate(clust):
    if c == 20:
        cv2.imshow('', image[i])
        cv2.waitKey()
'''

y = []
for test in test_data:
    if clust[test[0]] in numclust and clust[test[1]] in numclust:
        y.append(1)
    elif clust[test[0]] not in numclust and clust[test[1]] not in numclust:
        y.append(1)
    else:
        y.append(0)

savePrediction(y, pred_path)