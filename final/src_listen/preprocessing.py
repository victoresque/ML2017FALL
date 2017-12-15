import pickle
from gensim.models import KeyedVectors

with open('data/train.data', 'rb') as f:
    train_data = pickle.load(f)

with open('data/train.caption', 'r', encoding='utf-8') as f:
    train_caption = f.readlines()

fasttext = KeyedVectors.load_word2vec_format('fasttext/wiki.zh.vec')

print(fasttext['幹'])