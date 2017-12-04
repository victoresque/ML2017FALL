import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
lines, labels0 = readData('data/training_label.txt')
lines = []
with open('data/pre_training_nolabel.txt', 'r', encoding='utf_8') as f:
    for line in f:
        lines.append(line.split())
lines = lines[:400000]
print('  Converting...')
with open('data/pre_cmap.pkl', 'rb') as f:
    cmap = pickle.load(f)
cmapRefine(cmap)
transformByConversionMap(lines, cmap, iter=2)
print('  Padding lines...')
lines = padLines(lines, '_', maxlen)
print('  Transforming to word vectors...')
w2v = Word2Vec.load('data/word2vec.pkl')
transformByWord2Vec(lines, w2v)

print('Generating pseudo label...')
from keras.models import load_model

model = load_model('model.18-0.8532-0.8369.h5')
x_test = lines
y = model.predict(x_test, verbose=True).flatten()

lines = []
labels = []
for i, yi in enumerate(y):
    if i < 200000:
        lines.append(x_test[i])
        labels.append(labels0[i])
    else:
        if yi >= 0.95:
            lines.append(x_test[i])
            labels.append(1)
        elif yi <= 0.05:
            lines.append(x_test[i])
            labels.append(0)

print(len(lines))

print('  Splitting validation...')
x_valid, y_valid = lines[:20000], labels[:20000]
x_train, y_train = lines[20000:], labels[20000:]

print('Training...')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, History, ModelCheckpoint

model = Sequential()
model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,
               input_shape=(maxlen, 256)))
model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

history = History()
chkpoint = ModelCheckpoint('semi_model.{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5',
                           monitor='val_acc',
                           save_best_only=True,
                           save_weights_only=False)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[history, chkpoint],
          validation_data=[x_valid, y_valid])
model.save('rnn_0_semi.h5')
with open('result/semi_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)