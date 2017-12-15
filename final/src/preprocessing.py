# -*- coding: utf-8 -*-
import json
import jieba
import pickle
import numpy as np
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC
jieba.set_dictionary('dict/dict.txt.big')

w2v = Word2Vec.load('word2vec/zh.bin')
s2t, t2s = OpenCC('s2twp'), OpenCC('tw2sp')
def toW2V(s):
    offset = 0
    offsets = []
    for i, w in enumerate(s):
        ws = t2s.convert(w)
        wt = s2t.convert(w)
        if w in w2v.wv:
            s[i] = w2v.wv[w]
        elif ws in w2v.wv:
            s[i] = w2v.wv[ws]
        elif wt in w2v.wv:
            s[i] = w2v.wv[wt]
        else:
            s[i] = np.zeros((300, ))
        offsets.append(offset)
        offset += len(w)
    return s, offsets

with open('data/train-v1.1.json', 'r', encoding='utf_8') as f:
    train_data = json.load(f)

td = []
for data in tqdm(train_data['data']):
    t = {}
    for paragraph in data['paragraphs']:
        t['context'] = list(jieba.cut(paragraph['context']))
        t['context'], t['context_offset'] = toW2V(t['context'])
        for qa in paragraph['qas']:
            t['question'] = list(jieba.cut(qa['question']))
            t['question'], dummy = toW2V(t['question'])
            t['ans_st'] = int(qa['answers'][0]['answer_start'])
            t['ans_ed'] = t['ans_st'] + len(qa['answers'][0]['text'])
            td.append(t)

with open('data/train.pkl', 'wb') as f:
    pickle.dump(td, f)
