import string
import numpy as np

def removePunctuation(s):
    s = [''.join(c for c in w if c not in string.punctuation) for w in s]
    return [w for w in s if w]

def removeCommonWords(s):
    stopwords = set('is are be a an the to for of \
                    in will and at was were i he she 1 2 3 4 5 6 7 8 9 0'.split())
    s = [word if word not in stopwords else '' for word in s]
    return s

def getDictionaryAndTransform(lines):
    _dict = {}
    for line in lines:
        for word in line:
            if word not in _dict:
                _dict[word] = len(_dict) + 1
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            lines[i][j] = _dict[lines[i][j]]
    return lines, _dict

def readData(path, label=True):
    _lines, _labels = [], []
    _freq = {}
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            if label:
                _labels.append(int(line[0]))
                line = line[10:-1]
            else:
                line = line[:-1]
            line = line.split()
            line = removePunctuation(line)
            line = removeCommonWords(line)
            _lines.append(line)
            for word in line:
                if word in _freq:
                    _freq[word] = _freq[word] + 1
                else:
                    _freq[word] = 1

        for i in range(len(_lines)):
            line = _lines[i]
            for j in range(len(line)):
                if _freq[line[j]] < 8:
                    line[j] = ''

    # TODO: remove empty lines
    # TODO: check if all lines are in lower cases

    if label:
        return _lines, _labels
    else:
        return _lines

def readTestData(path):
    _lines = []
    with open(path, 'r', encoding='utf_8') as f:
        i = 0
        for line in f:
            if i:
                start = int(np.log10(max(1, i-1))) + 2
                line = line[start:].split()
                line = removePunctuation(line)
                line = removeCommonWords(line)
                _lines.append(line)
            i += 1
    return _lines

def transformByDictionary(lines, dictionary):
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j] in dictionary:
                lines[i][j] = dictionary[lines[i][j]]
            else:
                lines[i][j] = dictionary['']
    return lines