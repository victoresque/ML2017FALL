from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('data/word2vec')

print(model.wv.most_similar(positive=['電腦']))
print(model.wv.vocab['的'].count)
print(model.wv.vocab['電腦'].count)