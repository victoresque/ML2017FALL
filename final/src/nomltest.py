# -*- coding: utf-8 -*-
import json
import jieba
import pickle
from copy import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC
jieba.set_dictionary('dict/dict.txt.big')

with open('data/test-v1.1.json', 'r', encoding='utf_8') as f:
    train_data = json.load(f)

peak = 99999

td = []
for i, data in tqdm(enumerate(train_data['data'])):
    if i == peak:
        break
    t = {}
    for paragraph in data['paragraphs']:
        t['context'] = list(jieba.cut(paragraph['context']))
        for qa in paragraph['qas']:
            t['question'] = list(jieba.cut(qa['question']))
            t['id'] = qa['id']
            td.append(copy(t))

window = 35
answid = 25

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

aa = []
for i, t in tqdm(enumerate(td)):
    if i == peak:
        break
    context = list(''.join(t['context']))
    if al[i] < 0:
        al[i] = 0
    if ar[i] > len(context)+1:
        ar[i] = len(context)+1
    a = context[al[i]:ar[i]]
    q = list(''.join(t['question']))
    for j, ac in enumerate(a):
        if ac in q:
            a[j] = -1
        else:
            a[j] = al[i] + j
    a = [j for j in a if j != -1]
    if not a:
        a = [0]
    aa.append(a)

pd.DataFrame([[td[i]['id'], ' '.join([str(j) for j in aa[i]])] for i in range(len(aa))],
             columns=['id', 'answer']).to_csv('result/prediction.csv', index=False)
'''

pd.DataFrame([[td[i]['id'], ' '.join([str(i) for i in range(al[i], ar[i])])] for i in range(len(al))],
             columns=['id', 'answer']).to_csv('result/prediction.csv', index=False)

'''