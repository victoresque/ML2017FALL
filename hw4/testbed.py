import pickle
from gensim.models.word2vec import Word2Vec
from util import *

model = Word2Vec.load('data/word2vec.mdl')

lines = readData('data/pre_training_nolabel.txt', label=False)
dictionary = getDictionary(lines)
with open('data/pre_dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
print('Dictionary size:', len(dictionary))

model = Word2Vec(lines, size=128, min_count=1, workers=15)
model.save('data/word2vec.mdl')