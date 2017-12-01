import re
import string
import unicodedata as udata
import numpy as np
import pandas as pd

def getFrequencyDict(lines):
    freq = {}
    for s in lines:
        for w in s:
            if w in freq: freq[w] += 1
            else:         freq[w] = 1
    return freq

def removeAccents(lines):
    print('  Removing accents...')
    for i, s in enumerate(lines):
        s = [(''.join(c for c in udata.normalize('NFD', w) if udata.category(c) != 'Mn')) for w in s]
        lines[i] = [w for w in s if w]

def removePunctuations(lines):
    print('  Removing punctuations...')
    punctuations = '\"#$%&\'()*+,-/:;<=>@[\]_`{|}~'
    for i, s in enumerate(lines):
        s = [''.join(c for c in w if c not in punctuations) for w in s]
        lines[i] = [w for w in s if w]
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            excCnt, queCnt, dotCnt = w.count('!'), w.count('?'), w.count('.')
            if queCnt:        s[j] = '_?'
            elif dotCnt >= 3: s[j] = '_...'
            elif excCnt >= 5: s[j] = '_!!!'
            elif excCnt >= 3: s[j] = '_!!'
            elif excCnt >= 1: s[j] = '_!'
        lines[i] = s

def removeNotWords(lines):
    print('  Removing not words...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w[0] == '_': continue
            if w == '2':          s[j] = 'to'
            elif w.isnumeric():   s[j] = '_n'
            elif not w.isalpha(): s[j] = '_r'
        lines[i] = s

def convertTailDuplicates(lines):
    print('  Converting tail duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            s[j] = re.sub(r'(([a-dg-kmnp-rt-z])\2+)$', r'\g<2>', w)
        lines[i] = s

def convertHeadDuplicates(lines):
    print('  Converting head duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            s[j] = re.sub(r'^(([a-z])\2+)', r'\g<2>', w)
        lines[i] = s

def convertInlineDuplicates(lines, minfreq=64):
    print('  Converting inline duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if freq[w] > minfreq: continue
            w1 = re.sub(r'(([a-z])\2{2,})', r'\g<2>', w)  # repeated 3+ times, replace by 1
            w2 = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w) # repeated 3+ times, replace by 2
            w3 = re.sub(r'(([a-z])\2+)', r'\g<2>', w) # repeated 2+ times, replace by 1
            f0, f1, f2, f3 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0), freq.get(w3,0)
            fm = max(f0, f1, f2, f3)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1; 
            elif fm == f2: s[j] = w2; 
            else:          s[j] = w3; 
        lines[i] = s

def convertSlang(lines):
    print('  Converting slang...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == 'u': lines[i][j] = 'you'
            w1 = re.sub(r'in$', r'ing', w)
            w2 = re.sub(r'n$', r'ing', w)
            f0, f1, f2 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0)
            fm = max(f0, f1, f2)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            else:          s[j] = w2;
        lines[i] = s

def convertSingular(lines, minfreq=512):
    print('  Converting singular form...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if freq[w] > minfreq: continue
            w1 = re.sub(r's$', r'', w)
            w2 = re.sub(r'es$', r'', w)
            w3 = re.sub(r'ies$', r'y', w)
            f0, f1, f2, f3 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0), freq.get(w3,0)
            fm = max(f0, f1, f2, f3)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1; 
            elif fm == f2: s[j] = w2; 
            else:          s[j] = w3; 
        lines[i] = s

def convertRareWords(lines, minfreq=32):
    print('  Converting rare words...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if freq[w] < minfreq:
                s[j] = '_r'
        lines[i] = [w for w in s if w]

def convertCommonWords(lines):
    print('  Converting common words...')
    #beverbs = set('is was are were am s'.split())
    #articles = set('a an the'.split())
    #preps = set('to for of in at on by'.split())

    for i, s in enumerate(lines):
        #s = [word if word not in beverbs else '_b' for word in s]
        #s = [word if word not in articles else '_a' for word in s]
        #s = [word if word not in preps else '_p' for word in s]
        lines[i] = s

def convertPadding(lines):
    print('  Padding...')
    for i, s in enumerate(lines):
        lines[i] = ['_'] + s

def preprocessLines(lines):
    removeAccents(lines)
    removePunctuations(lines)
    removeNotWords(lines)
    convertTailDuplicates(lines)
    convertHeadDuplicates(lines)
    convertInlineDuplicates(lines, minfreq=len(lines)//5468)
    convertSlang(lines)
    convertSingular(lines, minfreq=len(lines)//2693)
    convertRareWords(lines, minfreq=len(lines)//21875)
    convertCommonWords(lines)
    convertPadding(lines)
    return lines

def readData(path, label=True):
    print('  Loading', path+'...')
    _lines, _labels = [], []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            if label:
                _labels.append(int(line[0]))
                line = line[10:-1]
            else:
                line = line[:-1]
            _lines.append(line.split())
    if label: return _lines, _labels
    else:     return _lines

def padLines(lines, value, maxlen):
    maxlinelen = 0
    for i, s in enumerate(lines):
        maxlinelen = max(len(s), maxlinelen)
    maxlinelen = max(maxlinelen, maxlen)
    for i, s in enumerate(lines):
        lines[i] = (['_'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
    return lines

def getDictionary(lines):
    _dict = {}
    for s in lines:
        for w in s:
            if w not in _dict:
                _dict[w] = len(_dict) + 1
    return _dict

def transformByDictionary(lines, dictionary):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in dictionary: lines[i][j] = dictionary[w]
            else:               lines[i][j] = dictionary['']

def transformByWord2Vec(lines, w2v):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_r']

def readTestData(path):
    _lines = []
    with open(path, 'r', encoding='utf_8') as f:
        for i, line in enumerate(f):
            if i:
                start = int(np.log10(max(1, i-1))) + 2
                _lines.append(line[start:].split())
    return _lines

def savePrediction(y, path, id_start=0):
    pd.DataFrame([[i+id_start, int(y[i])] for i in range(y.shape[0])],
                 columns=['id', 'label']).to_csv(path, index=False)