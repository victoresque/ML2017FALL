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
np.save('svd.S.npy', S)
del U
'''
U = np.load('svd.U.npy')
S = np.load('svd.S.npy')
print(S)
for i in range(4):
    print(S[i] / np.sum(S) * 100, '%')

def image_clip(x):
    x -= np.min(x)
    x /= np.max(x)
    x = (x * 255).astype(np.uint8)
    x = np.reshape(x, (600, 600, 3))
    return x

ids = [23, 96, 187, 253]
for id in ids:
    k = 4
    w = []
    y = X[id] - X_mean
    M = np.zeros(len(y))
    for i in range(k):
        eig = U[:, i]
        w.append(np.dot(y, eig))
        M += np.dot(y, eig) * eig
    M += X_mean
    M = image_clip(M)
    io.imsave('data/p1/{}.jpg'.format(id), M)
    io.imsave('data/p1/{}_.jpg'.format(id), np.reshape(X[id], (600, 600, 3)))

for i in range(4):
    io.imsave('data/p1/eig{}.jpg'.format(i), image_clip(-U[:, i]))

io.imsave('data/p1/mean.jpg', image_clip(X_mean))