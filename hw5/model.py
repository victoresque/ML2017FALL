from keras.layers import Embedding, Reshape, Dropout, Dense, dot, add, concatenate, Input, Flatten
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K

def MFModel(n_users, n_movies, d=64):
    user_input = Input(shape=(1, ))
    user_emb = Embedding(n_users, d)(user_input)
    user_emb = Flatten()(user_emb)

    movie_input = Input(shape=(1, ))
    movie_emb = Embedding(n_movies, d)(movie_input)
    movie_emb = Flatten()(movie_emb)

    user_bias = Embedding(n_users, 1)(user_input)
    user_bias = Flatten()(user_bias)

    movie_bias = Embedding(n_movies, 1)(movie_input)
    movie_bias = Flatten()(movie_bias)

    u = add([user_emb, user_bias])
    m = add([movie_emb, movie_bias])

    merged = dot([u, m], axes=1)
    output = add([merged, user_bias, movie_bias])

    return Model([user_input, movie_input], [output])

