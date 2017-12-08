import os
from gensim.models.word2vec import Word2Vec
from util import *
from param import *

def preprocessTestingData(path):
    print('Loading testing data...')
    lines = readTestData(path)

    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)
    lines = padLines(lines, '_', maxlen)

    w2v = Word2Vec.load(w2v_path)
    transformByWord2Vec(lines, w2v)
    return lines

def preprocessTrainingData(label_path, nolabel_path='', retrain=False, punctuation=True):
    print('Loading training data...')
    if retrain:
        preprocess(label_path, nolabel_path)

    lines, labels = readData(label_path)
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/corpus.txt')
    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    lines = readData(corpus_path, label=False)[:len(lines)]
    labels = np.array(labels)

    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)

    if not punctuation:
        removePunctuations(lines)

    lines = padLines(lines, '_', maxlen)
    w2v = Word2Vec.load(w2v_path)
    transformByWord2Vec(lines, w2v)
    return lines, labels

def preprocess(label_path, nolabel_path):
    print('Preprocessing...')
    labeled_lines, labels = readData(label_path)
    nolabel_lines = readData(nolabel_path, label=False)
    lines = labeled_lines + nolabel_lines

    lines, cmap = preprocessLines(lines)
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/corpus.txt')
    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    savePreprocessCorpus(lines, corpus_path)
    savePreprocessCmap(cmap, cmap_path)

    transformByConversionMap(lines, cmap)
    removeDuplicatedLines(lines)

    print('Training word2vec...')
    model = Word2Vec(lines, size=256, min_count=16, iter=16, workers=16)
    model.save(w2v_path)
