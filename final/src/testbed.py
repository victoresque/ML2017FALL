# -*- coding: utf-8 -*-
import json
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC

w2v = Word2Vec.load('word2vec/zh.bin')

for i, key in enumerate(w2v.wv.vocab):
    print(key)

print(w2v.wv.most_similar(['。']))
with open('../data/qa/Delta_Chinese_RC_dataset_2art5parag.json', 'r', encoding='utf_8') as f:
    train_data = json.load(f)

opencc = OpenCC('s2twp')
for data in train_data['data']:
    data['title'] = opencc.convert(data['title'])
    print(data['title'])
    for paragraph in data['paragraphs']:
        print('  ', paragraph['context'])
        for qa in paragraph['qas']:
            print('    ', qa['question'])
            print('      ', qa['answers'][0]['text'])
        print('')
