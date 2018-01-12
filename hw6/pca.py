import os
import sys
import numpy as np
from skimage import io

images_path = sys.argv[1]
target_name = sys.argv[2]

X = []

filenames = os.listdir(images_path)
for filename in filenames:
    img = io.imread(os.path.join(images_path, filename))
    X.append(img.flatten())

X = np.array(X)
X_mean = np.mean(X, axis=0)

U, S, V = np.linalg.svd((X - X_mean).T, full_matrices=False)

def image_clip(x):
    x -= np.min(x)
    x /= np.max(x)
    x = (x * 255).astype(np.uint8)
    x = np.reshape(x, (600, 600, 3))
    return x

target = io.imread(os.path.join(images_path, target_name))

k = 4
y = target.flatten() - X_mean
M = np.zeros(len(y))
for i in range(k):
    eig = U[:, i]
    M += np.dot(y, eig) * eig
M += X_mean
M = image_clip(M)

io.imsave('reconstruction.jpg', M)