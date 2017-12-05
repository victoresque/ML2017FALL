import pickle
import string
import pandas as pd
from gensim.models.word2vec import Word2Vec
from util import *

lines = [['a', 'b', 'c'], ['a', 'b', 'a'], ['a', 'b', 'c']]
print(lines)
removeDuplicatedLines(lines)
print(lines)

d = {'1':1, '2':2}
for key, value in d.items():
    print(key, value)

with open('data/pre_cmap.pkl', 'rb') as f:
    cmap = pickle.load(f)
print(cmap['_r'])
cmapRefine(cmap)
print(cmap['go2'])

'''
lines = []
with open('data/pre_training_nolabel.txt', 'r', encoding='utf_8') as f:
    for line in f:
        lines.append(line.split())
maxlinelen = 0
for line in lines:
    if len(line) == 229:
        print(line)
    maxlinelen = max(maxlinelen, len(line))
print(maxlinelen)

lines, labels = readData('data/training_label.txt')
c0, c1 = 0, 0
for i in labels:
    if i: c1 += 1
    else: c0 += 1

print(c0, c1)
'''

model = Word2Vec.load('data/word2vec.pkl')

print(model.wv.most_similar(positive='_!'.split(), negative=''.split()))
print(model.wv.most_similar(positive='love'.split(), negative=''.split()))
print(model.wv.most_similar(positive='like'.split(), negative=''.split()))

'''
print('Training word2vec...')
lines = readData('data/pre_training_nolabel.txt', label=False)
model = Word2Vec(lines, size=128, min_count=1, workers=15)
model.save('data/word2vec.mdl')
'''


