from keras.layers import Embedding, Reshape, Dropout, Dense, dot, add, concatenate
from keras.layers import Input, Flatten, GRU
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K

def QA(p_maxlen, q_maxlen):
    input1 = Input(shape=(p_maxlen, 300))
    input2 = Input(shape=(q_maxlen, 300))
    P = GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(input1)
    P = GRU(512, dropout=0.5, recurrent_dropout=0.5)(P)
    #P = GRU(512, dropout=0.5, recurrent_dropout=0.5)(input1)
    Q = GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(input2)
    Q = GRU(512, dropout=0.5, recurrent_dropout=0.5)(Q)
    #Q = GRU(512, dropout=0.5, recurrent_dropout=0.5)(input2)
    R = concatenate([P, Q])
    R1 = Dense(512, activation='relu')(R)
    R1 = Dense(512, activation='relu')(R1)
    output1 = Dense(1, activation='relu')(R1)
    R2 = Dense(512, activation='relu')(R)
    R2 = Dense(512, activation='relu')(R2)
    output2 = Dense(1, activation='relu')(R2)
    return Model([input1, input2], [output1, output2])
