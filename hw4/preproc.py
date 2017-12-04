import pickle
from gensim.models.word2vec import Word2Vec
from util import *
from param import *
'''
print('Loading data...')
labeled_lines, labels = readData('data/training_label.txt')
nolabel_lines = readData('data/training_nolabel.txt', label=False)
testing_lines = readTestData('data/testing_data.txt')
lines = labeled_lines + nolabel_lines + testing_lines

print('Preprocessing data...')
lines, cmap = preprocessLines(lines)
with open('data/pre_corpus.txt', 'w', encoding='utf_8') as f:
    for line in lines:
        f.write(' '.join(line)+'\n')
with open('data/pre_cmap.pkl', 'wb') as f:
    pickle.dump(cmap, f)
'''
print('Corpus refinement...')
lines = []
with open('data/pre_corpus.txt', 'r', encoding='utf_8') as f:
    for line in f:
        lines.append(line.split())
with open('data/pre_cmap.pkl', 'rb') as f:
    cmap = pickle.load(f)
cmapRefine(cmap)
transformByConversionMap(lines, cmap)
with open('data/pre_corpus_refined.txt', 'w', encoding='utf_8') as f:
    for line in lines:
        f.write(' '.join(line)+'\n')

print('Training word2vec...')
model = Word2Vec(lines, size=256, min_count=16, iter=16, workers=16)
model.save('data/word2vec.pkl')
