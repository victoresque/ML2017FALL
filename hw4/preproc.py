from gensim.models.word2vec import Word2Vec
from util import *
from param import *

def loadTestingData(path):
    print('Loading testing data...')
    lines = readTestData(path)

    print('Preprocessing...')
    cmap = loadPreprocessCmap('model/pre_cmap.pkl')
    transformByConversionMap(lines, cmap)
    lines = padLines(lines, '_', maxlen)

    w2v = Word2Vec.load('model/word2vec.pkl')
    transformByWord2Vec(lines, w2v)

def loadTrainingData(label_path, nolabel_path):
    print('Loading training data...')
    preprocess(label_path, nolabel_path)

    lines, labels = readData(label_path)
    lines = readData('model/pre_corpus.txt', label=False)[:len(lines)]
    labels = np.array(labels)

    cmap = loadPreprocessCmap('model/pre_cmap.pkl')
    transformByConversionMap(lines, cmap)
    lines = padLines(lines, '_', maxlen)

    w2v = Word2Vec.load('model/word2vec.pkl')
    transformByWord2Vec(lines, w2v)
    return lines, labels

def preprocess(label_path, nolabel_path):
    print('Preprocessing...')
    labeled_lines, labels = readData(label_path)
    nolabel_lines = readData(nolabel_path, label=False)
    lines = labeled_lines + nolabel_lines

    lines, cmap = preprocessLines(lines)
    savePreprocessCorpus(lines, 'model/pre_corpus.txt')
    savePreprocessCmap(cmap, 'model/pre_cmap.pkl')

    transformByConversionMap(lines, cmap)
    savePreprocessCorpus(lines, 'model/pre_corpus_refined.txt')
    removeDuplicatedLines(lines)

    print('Training word2vec...')
    model = Word2Vec(lines, size=256, min_count=16, iter=16, workers=16)
    model.save('model/word2vec.pkl')
