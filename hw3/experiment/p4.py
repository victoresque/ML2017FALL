import numpy as np
import pandas as pd
import keras
import os
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from param import *

model = load_model('../model/0.67623.h5')

id = 987
X_test = np.load('../data/X_test.npy') / 255
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)
img = X_test[id]
plt.figure()
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.show()

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
img = X_test[id].reshape(1, img_rows, img_cols, 1)

input_img = model.input
pred = model.predict(img).argmax(axis=-1)
target = K.mean(model.output[:, pred])
grads = K.gradients(target, input_img)[0]

layer_dict = dict([(layer.name, layer) for layer in model.layers])
target_layer = layer_dict['dense_3']
fn = K.function([input_img, K.learning_phase()], [grads])

[heatmap] = fn([img, target_layer])
heatmap = heatmap.reshape(48, 48)
heatmap = np.abs(heatmap)
heatmap = heatmap / np.max(heatmap)

thres = 0.1
see = img.reshape(48, 48)
see[np.where(heatmap <= thres)] = np.mean(see)

plt.figure()
plt.imshow(heatmap, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(see,cmap='gray')
plt.tight_layout()
plt.show()