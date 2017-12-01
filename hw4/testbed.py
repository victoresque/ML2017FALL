import pickle
import string
from gensim.models.word2vec import Word2Vec
from util import *

model = Word2Vec.load('data/word2vec.pkl')
print(model.wv.most_similar(positive='love like'.split(), negative='hate'.split()))
print(model.wv.most_similar(positive='delicious'.split(), negative=''.split()))
print(model.wv.doesnt_match('tasty delicious yum yummy good'.split()))

'''
print('Training word2vec...')
lines = readData('data/pre_training_nolabel.txt', label=False)
model = Word2Vec(lines, size=128, min_count=1, workers=15)
model.save('data/word2vec.mdl')
'''


