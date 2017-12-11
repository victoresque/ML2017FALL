from keras.layers import Embedding, Reshape, Dropout, Dense, dot, add, concatenate, Input, Flatten
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K

def MFModel(n_users, n_movies, d):
    input1 = Input(shape=(1, ))
    P = Embedding(n_users, d, input_length=1)(input1)
    P = Flatten()(P)
    input2 = Input(shape=(1, ))
    Q = Embedding(n_movies, d, input_length=1)(input2)
    Q = Flatten()(Q)
    R = dot([P, Q], axes=1)

    return Model([input1, input2], R)

