import numpy as np
from skimage import io

X = []
for i in range(415):
    img = io.imread('data/faces/{}.jpg'.format(i))
    X.append(img.flatten())

X = np.array(X)
X_mean = np.mean(X, axis=0)
'''
U, S, V = np.linalg.svd((X - X_mean).T, full_matrices=False)
np.save('svd.U.npy', U)
del U
'''
U = np.load('svd.U.npy')

k = 256
y = X[7] - X_mean
M = np.zeros(len(y))
for i in range(k):
    eig = U[:, i]
    M += np.dot(y, eig) * eig

M += X_mean
M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)
M = np.reshape(M, (600, 600, 3))

import cv2
cv2.imshow('', cv2.cvtColor(M, cv2.COLOR_BGR2RGB))
cv2.waitKey()

del M

import cv2
M = -U[:, 9]
M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)
M = np.reshape(M, (600, 600, 3))
cv2.imshow('', cv2.cvtColor(M, cv2.COLOR_BGR2RGB))
cv2.waitKey()