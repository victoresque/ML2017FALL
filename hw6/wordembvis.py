import re
import numpy as np
import matplotlib
matplotlib.rcParams['font.family']=["WenQuanYi Zen Hei"]
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 9]
from adjustText import adjust_text
from gensim.models.word2vec import Word2Vec

def plot(Xs, Ys, Texts):
    fig, ax = plt.subplots()
    ax.plot(Xs, Ys, 'o')
    texts = [plt.text(X, Y, Text) for X, Y, Text in zip(Xs, Ys, Texts)]
    adjust_text(texts, Xs, Ys,
                arrowprops=dict(arrowstyle='->', color='red'),
                autoalign='x')
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [re.sub('−', '-', i) for i in labels]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [re.sub('−', '-', i) for i in labels]
    ax.set_yticklabels(labels)
    plt.show()

X = []
T = []
w2v = Word2Vec.load('data/word2vec')
for word in w2v.wv.vocab:
    if 8000 >= w2v.wv.vocab[word].count >= 2000:
        X.append(w2v.wv[word])
        T.append(word)

from sklearn.manifold import TSNE

X = TSNE(n_components=2, n_iter=20000, verbose=1).fit_transform(X)

plot(X[:, 0], X[:, 1], T)