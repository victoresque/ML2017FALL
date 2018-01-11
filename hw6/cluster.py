import numpy as np

image = np.load('data/image.npy').astype(np.float32)
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

image = encoder.predict(image, batch_size=256, verbose=1)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# pca = PCA(n_components=4).fit(image[:100])
# print('PCA fitting done')
# image = pca.transform(image)
# image = TSNE(n_components=2, verbose=1).fit_transform(image)
# np.save('data/tsne.npy', image)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20, verbose=1).fit(image)
clust = kmeans.predict(image)

np.save('data/clust.npy', clust)