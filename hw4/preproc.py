from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
labeled_lines, labels = readData('data/training_label.txt')
nolabel_lines = readData('data/training_nolabel.txt', label=False)

lines = labeled_lines + nolabel_lines
print(len(lines))

print('Preprocessing data...')
lines = preprocessLines(lines)
with open('data/pre_training_nolabel.txt', 'w', encoding='utf_8') as f:
    for line in lines:
        f.write(' '.join(line)+'\n')


