import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

def getLineset(lines):
    lineset = set({})
    for line in lines:
        lineset.add(' '.join(line))
    return lineset

def dealWithIt(lines, lineset):
    for i, s in enumerate(lines):
        s_str = ' '.join(s)
        if i < 200000:
            if s_str in lineset:
                lineset.remove(s_str)
        else:
            if s_str in lineset:
                lineset.remove(s_str)
            else:
                lines[i], lines[-1] = lines[-1], lines[i]
                del lines[-1]

print('Loading data...')
lines0, labels0 = readData('data/training_label.txt')
lines = readData('data/pre_corpus_refined.txt', label=False)[:600000]
print('Converting...')
lineset = getLineset(lines)
cmap = loadPreprocessCmap('data/pre_cmap.pkl')
transformByConversionMap(lines, cmap)
dealWithIt(lines, lineset)
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
print('  Transforming to word vectors...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)
x_test = lines
'''
print('Generating pseudo label...')
from keras.models import load_model
model = load_model('model.58-0.8406-0.8385.h5')
y = model.predict(x_test, batch_size=256, verbose=True).flatten()
with open('data/semi_y.pkl', 'wb') as f:
    pickle.dump(y, f)
'''

with open('data/semi_y.pkl', 'rb') as f:
    y = pickle.load(f)

lines = []
labels = []
for i, yi in enumerate(y):
    if i > 600000:
        break
    if i < 200000:
        lines.append(x_test[i])
        labels.append(labels0[i])
    else:
        if yi >= 0.75 and yi <= 0.9:
            lines.append(x_test[i])
            labels.append(1)
        elif yi <= 0.25 and yi >= 0.1:
            lines.append(x_test[i])
            labels.append(0)
print(len(lines))

print('  Splitting validation...')
#shuffleData(lines, labels)
v = 20000
x_valid, y_valid = lines[:v], labels[:v]
x_train, y_train = lines[v:], labels[v:]

print('Training...')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU
from keras.callbacks import EarlyStopping, History, ModelCheckpoint

model = Sequential()
model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,
               input_shape=(maxlen, 256)))
model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(512, activation='selu'))
model.add(Dense(512, activation='selu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()
history = History()
chkpoint = ModelCheckpoint('semi_model.{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5',
                           monitor='val_acc',
                           save_best_only=False,
                           save_weights_only=False)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[history, chkpoint],
          validation_data=[x_valid, y_valid])
with open('result/semi_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)