'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from scipy.misc import imsave
import numpy as np
import time
from keras.models import load_model
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt

img_width, img_height = 48, 48

layer_name = 'conv2d_4'
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

model = load_model('../model/0.67623.h5')
id = 6999
X_test = np.load('../data/X_test.npy') / 255
X_test = X_test.reshape(X_test.shape[0], img_width, img_height)
img = X_test[id]

model.summary()
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

filters = []
for filter_index in range(64):
    print('Processing filter %d' % filter_index)
    start_time = time.time()
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)
    iterate = K.function([input_img], [loss, grads])
    step = 1.
    input_img_data = np.random.random((1, img_width, img_height, 1))

    for i in range(64):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        print('Current loss value:', loss_value)

    img = deprocess_image(input_img_data[0])
    filters.append(img)
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

filters = np.array(filters)
print(filters.shape)
fig = plt.figure(figsize=(14,8))
nb_filter = filters.shape[0]
print(nb_filter)
for i in range(nb_filter):
    ax = fig.add_subplot(nb_filter/16,16,i+1)
    ax.imshow(filters[i,:,:,0],cmap='PuBuGn')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
fig.suptitle('Filters of layer {} (# Ascent Epoch 64)'.format('conv_4'))
plt.tight_layout()
plt.show()


collect_layers = list()
collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict[layer_name].output]))
photo = X_test[id]
for cnt, fn in enumerate(collect_layers):
    im = fn([photo.reshape(1,48,48,1),0])
    fig = plt.figure(figsize=(14,8))
    nb_filter = im[0].shape[3]
    for i in range(64):
        ax = fig.add_subplot(nb_filter/16,16,i+1)
        ax.imshow(im[0][0,:,:,i],cmap='PuBuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer {} (Given image{})'.format('conv_4', id))
    plt.tight_layout()
    plt.show()

