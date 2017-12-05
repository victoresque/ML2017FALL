import pickle
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

print('Loading data...')
labeled_lines, labels = readData('data/training_label.txt')
nolabel_lines = readData('data/training_nolabel.txt', label=False)
testing_lines = readTestData('data/testing_data.txt')
lines = labeled_lines + nolabel_lines + testing_lines

print('Preprocessing data...')
lines, cmap = preprocessLines(lines)
savePreprocessCorpus(lines, 'data/pre_corpus.txt')
savePreprocessCmap(cmap, 'data/pre_cmap.pkl')

print('Corpus refinement...')
lines = loadPreprocessCorpus('data/pre_corpus.txt')
cmap = loadPreprocessCmap('data/pre_cmap.pkl')
transformByConversionMap(lines, cmap)
savePreprocessCorpus(lines, 'data/pre_corpus_refined.txt')
removeDuplicatedLines(lines)

print('Training word2vec...')
model = Word2Vec(lines, size=256, min_count=16, iter=16, workers=16)
model.save('data/word2vec.pkl')
