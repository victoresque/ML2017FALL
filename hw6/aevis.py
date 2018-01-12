import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams["figure.figsize"] = [8, 6]
'''
image = np.load('data/visualization.npy').astype(np.float32)
image = np.reshape(image, (-1, 28, 28, 1))
image = image / 255
image0 = image[:]

from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Dense, Reshape
from keras.models import Model, load_model

image = np.reshape(image, (-1, 28*28))
input = Input((28*28,))
x = Dense(128, activation='selu')(input)
x = Dense(64, activation='selu')(x)
encoded = Dense(32, activation='selu')(x)
x = Dense(64, activation='selu')(encoded)
x = Dense(128, activation='selu')(x)
output = Dense(28*28, activation='selu')(x)

encoder = Model(input, encoded)
encoder.load_weights('ae32.h5', by_name=True)

X = encoder.predict(image, batch_size=256, verbose=1)
np.save('data/aevis.npy', X)
'''

# X = np.load('data/aevis.npy')
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=20, verbose=1).fit(X)
# label = kmeans.predict(X)
# np.save('data/aevis_kmeans.npy', label)

# from sklearn.manifold import TSNE
# X = TSNE(n_components=2, n_iter=1000, verbose=1).fit_transform(X)
# np.save('data/aevis_tsne.npy', X)

X = np.load('data/aevis_tsne.npy')

cmap0 = matplotlib.cm.get_cmap('YlGnBu')
cmap1 = matplotlib.cm.get_cmap('YlOrRd')

ax =  plt.subplots()[1]
ax.scatter(X[5000:, 0], X[5000:, 1], marker='.', color=cmap0(0.5))
ax.scatter(X[:5000, 0], X[:5000, 1], marker='.', color=cmap1(0.5))

label = np.load('data/aevis_kmeans.npy')
Xc = []
for i in range(20):
    Xc.append(np.array([x for j, x in enumerate(X) if label[j]==i]))

ax = plt.subplots()[1]
numclust = {0, 2, 3, 5, 7, 8, 9, 12, 14, 16, 18}
for i in range(20):
    if i in numclust:
        ax.scatter(Xc[i][:, 0], Xc[i][:, 1], marker='.', color=cmap0(i / 30 + 0.3))
    else:
        ax.scatter(Xc[i][:, 0], Xc[i][:, 1], marker='.', color=cmap1(i / 30 + 0.3))


plt.show()


