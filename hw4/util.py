import string
import re
import unicodedata as udata
import numpy as np
import pandas as pd

def removeAccents(lines):
    print('  Removing accents...')
    for i, s in enumerate(lines):
        s = [(''.join(c for c in udata.normalize('NFD', w) if udata.category(c) != 'Mn')) for w in s]
        lines[i] = [w for w in s if w]
    return lines
def removePunctuations(lines):
    print('  Removing punctuations...')
    for i, s in enumerate(lines):
        s = [''.join(c for c in w if c not in string.punctuation) for w in s]
        lines[i] = [w for w in s if w]
    return lines
def removeNotWords(lines):
    print('  Removing not words...')
    for i, s in enumerate(lines):
        s = [w for w in s if w.isalpha()]
        lines[i] = [w for w in s if w]
    return lines
def getFrequencyDict(lines):
    freq = {}
    for line in lines:
        for word in line:
            if word in freq: freq[word] += 1
            else:            freq[word] = 1
    return freq
def convertTailDuplicates(lines):
    print('  Converting tail duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if freq[w] > 32: continue
            s[j] = re.sub(r'(([a-z])\2{2,})$', r'\g<2>', w, 1)
        lines[i] = s
    return lines
def convertHeadDuplicates(lines):
    print('  Converting head duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if freq[w] > 32: continue
            s[j] = re.sub(r'^(([a-z])\2{2,})', r'\g<2>', w)
        lines[i] = s
    return lines
def convertInlineDuplicates(lines):
    print('  Converting inline duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if freq[w] > 32: continue
            w1 = re.sub(r'(([a-z])\2{2,})', r'\g<2>', w)  # repeated 3+ times, replace by 1
            w2 = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w) # repeated 3+ times, replace by 2
            w3 = re.sub(r'(([a-z])\2{1,})', r'\g<2>', w) # repeated 2+ times, replace by 1
            f0, f1, f2, f3 = freq[w], freq[w1], freq[w2], freq[w3]
            fm = max(f0, f1, f2, f3)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1; 
            elif fm == f2: s[j] = w2;
            else:          s[j] = w3;
        lines[i] = s
    return lines

def convertSameWords(lines):
    lines = convertTailDuplicates(lines)
    lines = convertHeadDuplicates(lines)
    lines = convertInlineDuplicates(lines)
    return lines

def convertRareWords(s, freq, minfreq=8):
    for i in range(len(s)):
        if freq[s[i]] < minfreq:
            s[i] = ''
    return s

def convertCommonWords(s):
    stopwords = set('is are am be a an the to for of '
                    'in will and at was were i he she'.split())
    s = [word if word not in stopwords else '' for word in s]
    return s

def preprocessLines(lines):
    lines = removeAccents(lines)
    lines = removePunctuations(lines)
    lines = removeNotWords(lines)
    lines = convertSameWords(lines)
    print(lines[60011])

    freq = getFrequencyDict(lines)
    for i in range(len(lines)):
        lines[i] = convertRareWords(lines[i], freq, minfreq=8)
        lines[i] = convertCommonWords(lines[i])
    return lines

# TODO: remove empty lines

def readData(path, label=True):
    print('  Loading data...')
    _lines, _labels = [], []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            _labels.append(int(line[0]))
            if label: line = line[10:-1]
            else:     line = line[:-1]
            _lines.append(line.split())
    if label: return _lines, _labels
    else:     return _lines

def getDictionary(lines):
    _dict = {}
    for line in lines:
        for word in line:
            if word not in _dict:
                _dict[word] = len(_dict) + 1
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            lines[i][j] = _dict[lines[i][j]]
    return _dict

def transformByDictionary(lines, dictionary):
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j] in dictionary:
                lines[i][j] = dictionary[lines[i][j]]
            else:
                lines[i][j] = dictionary['']
    return lines

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