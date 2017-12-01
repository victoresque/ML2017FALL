import pickle
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
labeled_lines, labels = readData('data/training_label.txt')
nolabel_lines = readData('data/training_nolabel.txt', label=False)
lines = labeled_lines + nolabel_lines

print('Preprocessing data...')
lines = preprocessLines(lines)
with open('data/pre_training_nolabel.txt', 'w', encoding='utf_8') as f:
    for line in lines:
        f.write(' '.join(line)+'\n')

print('Generating dictionary...')
lines = readData('data/pre_training_nolabel.txt', label=False)
dictionary = getDictionary(lines)
with open('data/pre_dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
print('Dictionary size:', len(dictionary))

lines = []
with open('data/pre_training_nolabel.txt', 'r', encoding='utf_8') as f:
    for line in f:
        lines.append(line.split())

print('Training word2vec...')
model = Word2Vec(lines, size=128, min_count=1, iter=16, workers=16)
model.save('data/word2vec.pkl')