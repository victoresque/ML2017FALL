# -*- coding: utf-8 -*-
import json
import jieba
import pickle
from copy import copy
import numpy as np
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC
jieba.set_dictionary('dict/dict.txt.big')

w2v = Word2Vec.load('word2vec/zh.bin')

with open('data/train-v1.1.json', 'r', encoding='utf_8') as f:
    train_data = json.load(f)

peak = 10

td = []
for i, data in tqdm(enumerate(train_data['data'])):
    if i == peak:
        break
    t = {}
    for paragraph in data['paragraphs']:
        t['context'] = list(jieba.cut(paragraph['context']))
        for qa in paragraph['qas']:
            t['question'] = list(jieba.cut(qa['question']))
            t['ans_st'] = int(qa['answers'][0]['answer_start'])
            t['ans_ed'] = t['ans_st'] + len(qa['answers'][0]['text'])
            t['id'] = qa['id']
            td.append(copy(t))

window = 24
answid = 20

al = []
ar = []
for i, t in tqdm(enumerate(td)):
    if i == peak:
        break
    p = ''.join(t['context'])
    q = ''.join(t['question'])
    maxpsegcnt = 0
    psegleft = 1000
    psegright = 0
    for left in range(0, len(p)-window):
        right = left + window # noninclusive
        pseg = set(p[left:right])
        q = set(q)
        psegcnt = 0
        for wq in q:
            for wp in pseg:
                if wq == wp and not wq.isnumeric():
                    psegcnt += 1
        if psegcnt >= maxpsegcnt:
            psegleft = left
            psegright = right
        maxpsegcnt = max(maxpsegcnt, psegcnt)
    psegleftid = 0
    psegrightid = 0
    for i in range(psegleft):
        psegleftid += len(p[i])
    for i in range(psegright):
        psegrightid += len(p[i])

    psegmidid = ( psegleftid * 3 + psegrightid * 2 ) // 5
    al.append(max(0, psegmidid-answid))
    ar.append(min(len(p), psegmidid))

for i in range(20):
    if i == peak:
        break
    p = ''.join(td[i]['context'])
    q = ''.join(td[i]['question'])
    print('---', q, '---')
    print('-------', p[al[i]:ar[i]], '-------')
    print('-------', p[td[i]['ans_st']:td[i]['ans_ed']], '-------')
    print('')

import pandas as pd
pd.DataFrame([[td[i]['id'], ' '.join([str(i) for i in range(al[i], ar[i])])   ] for i in range(len(al))],
                 columns=['id', 'answer_text']).to_csv('result/prediction.csv', index=False)