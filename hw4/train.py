import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
lines, labels = readData('data/training_label.txt')
lines = readData('data/pre_corpus.txt', label=False)[:len(lines)]
shuffleData(lines, labels)

print('Preprocessing data...')
print('  Converting...')
cmap = loadPreprocessCmap('data/pre_cmap.pkl')
transformByConversionMap(lines, cmap)
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
labels = np.array(labels)
print('  Transforming to word vector...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)
x_train, y_train = lines, labels

print('Training...')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, History, ModelCheckpoint

history = History()
earlystop = EarlyStopping(patience=3)

def train(id):
    model = Sequential()
    model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,
                  input_shape=(maxlen, 256)))
    model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(512, activation='selu'))
    model.add(Dense(512, activation='selu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    chkpoint = ModelCheckpoint('model/m'+str(id)+'.{epoch:02d}-{acc:.3f}-{val_acc:.3f}.h5', save_best_only=False, save_weights_only=False)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
              callbacks=[history, chkpoint])

for i in range(8):
    train(i+1)


'''
all m0

m0: GRU 512 0.5 0.5 - GRU 512 0.5 0.5 - Dense 512 selu - Dense 512 selu
m1: LSTM 512 0.5 0.5 - LSTM 512 0.5 0.5 - Dense 512 selu - Dense 512 selu 
m2: GRU 512 0.5 - GRU 512 0.5 - Dense 512 selu - Dense 512 selu
m3: LSTM 256 0.5 0.2 - LSTM 256 0.5 0.2 - Dense 512 selu - Dropout 0.2
m4: GRU 256 0.5 0.2 - GRU 256 0.5 0.2 - Dense 512 selu - Dropout 0.2
'''