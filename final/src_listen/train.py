import pickle
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
from param import *

with open('data/train.data', 'rb') as f:
    train_data = pickle.load(f)
with open('data/train.caption', 'r', encoding='utf-8') as f:
    train_caption = f.readlines()
fasttext = KeyedVectors.load_word2vec_format('fasttext/wiki.zh.vec')

train_caption = [s.split() for s in train_caption]
train_caption = [[fasttext[w] for w in s if w in fasttext] for s in train_caption]

for i, data in tqdm(enumerate(train_data)):
    train_data[i] = np.append([np.zeros((39,))] * (INPUT_MAX_LEN - len(data)), data, axis=0)
for i, caption in tqdm(enumerate(train_caption)):
    train_caption[i] = caption + [np.zeros((300,))] * (OUTPUT_MAX_LEN - len(caption))

from keras.layers import GRU, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers.core import Dense, Reshape, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(LSTM(hidden_size, input_shape=(INPUT_MAX_LEN, 39), return_sequences=False))
model.add(Dense(hidden_size, activation="relu"))
model.add(RepeatVector(OUTPUT_MAX_LEN))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(300, activation="linear")))
model.compile(loss="hinge", optimizer='adam')

model.summary()

train_x = np.array(train_data)
train_y = np.array(train_caption)

chkpoint = ModelCheckpoint('model{epoch:02d}-{loss:.4f}-{val_loss:.4f}.h5', save_best_only=False)
model.fit(train_x, train_y, validation_split=0.1, epochs=100, batch_size=256,  callbacks=[chkpoint])