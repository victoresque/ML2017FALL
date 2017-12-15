import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from param import *
from gensim.models import KeyedVectors
'''
test_options = pd.read_csv('data/test.csv', header=None).values

fasttext = KeyedVectors.load_word2vec_format('fasttext/wiki.zh.vec')
test_options = [[s.split() for s in test] for test in test_options]
#test_options = [[[fasttext[w] for w in s if w in fasttext] for s in test] for test in test_options]
for i, test in tqdm(enumerate(test_options)):
    for j, s in enumerate(test):
        for k, w in enumerate(s):
            if w in fasttext:
                test_options[i][j][k] = fasttext[w]
            else:
                test_options[i][j][k] = np.zeros((300,))


np.save('result/test_options.npy', test_options)
'''
test_options = np.load('result/test_options.npy')

def savePrediction(y, path, id_start=1):
    pd.DataFrame([[i+id_start, int(y[i])] for i in range(len(y))],
                 columns=['id', 'answer']).to_csv(path, index=False)

def score(opt1, opt2):
    _score = 0
    for i in range(len(opt1)):
        _score += cosine_similarity(opt1[i], opt2[i])

    return _score
y = []
for i, test in tqdm(enumerate(test_options)):
    scores = []

    scores.append(score(test[0][0], test[1][0]))
    scores.append(score(test[0][0], test[2][0]))
    scores.append(score(test[0][0], test[3][0]))
    scores.append(score(test[1][0], test[2][0]))
    scores.append(score(test[1][0], test[3][0]))
    scores.append(score(test[2][0], test[3][0]))

    if np.argmax(scores) == 0:
        if scores[1]+scores[2] > scores[3]+scores[4]:
            y.append(0)
        else:
            y.append(1)
    elif np.argmax(scores) == 1:
        if scores[0]+scores[2] > scores[3]+scores[5]:
            y.append(0)
        else:
            y.append(2)
    elif np.argmax(scores) == 2:
        if scores[0] + scores[1] > scores[4] + scores[5]:
            y.append(0)
        else:
            y.append(3)
    elif np.argmax(scores) == 3:
        if scores[0] + scores[4] > scores[1] + scores[5]:
            y.append(1)
        else:
            y.append(2)
    elif np.argmax(scores) == 4:
        if scores[0] + scores[3] > scores[2] + scores[5]:
            y.append(1)
        else:
            y.append(3)
    elif np.argmax(scores) == 5:
        if scores[1] + scores[3] > scores[2] + scores[4]:
            y.append(2)
        else:
            y.append(3)
    # print(y)


savePrediction(y, 'result/prediction.csv')