import pickle
import string
import pandas as pd
from gensim.models.word2vec import Word2Vec
from util import *

lines, labels = readData('data/training_label.txt')
c0, c1 = 0, 0
for i in labels:
    if i: c1 += 1
    else: c0 += 1

print(c0, c1)

model = Word2Vec.load('data/word2vec.pkl')

print(model.wv.most_similar(positive='_!'.split(), negative=''.split()))
print(model.wv.most_similar(positive='love'.split(), negative=''.split()))
print(model.wv.most_similar(positive='microsoft'.split(), negative=''.split()))
print(model.wv.most_similar(positive='apple'.split(), negative=''.split()))
print(model.wv.most_similar(positive='windows'.split(), negative=''.split()))
print(model.wv.most_similar(positive='king'.split(), negative=''.split()))
print(model.wv.most_similar(positive='delicious'.split(), negative=''.split()))
print(model.wv.doesnt_match('tasty delicious yum yummy good'.split()))

'''
print('Training word2vec...')
lines = readData('data/pre_training_nolabel.txt', label=False)
model = Word2Vec(lines, size=128, min_count=1, workers=15)
model.save('data/word2vec.mdl')
'''


