import re
import string
import unicodedata as udata
import numpy as np
import pandas as pd

original_lines = []

def getFrequencyDict(lines):
    freq = {}
    for s in lines:
        for w in s:
            if w in freq: freq[w] += 1
            else:         freq[w] = 1
    return freq

def initializeCmap(lines):
    print('  Initializing conversion map...')
    cmap = {}
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            cmap[w] = w
    print('    Conversion map size:', len(cmap))
    return cmap

def convertAccents(lines, cmap):
    print('  Converting accents...')
    for i, s in enumerate(lines):
        s = [(''.join(c for c in udata.normalize('NFD', w) if udata.category(c) != 'Mn')) for w in s]
        for j, w in enumerate(s):
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s
    clist = 'abcdefghijklmnopqrstuvwxyz0123456789.!?'
    for i, s in enumerate(lines):
        s = [''.join([c for c in w if c in clist]) for w in s]
        for j, w in enumerate(s):
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertPunctuations(lines, cmap):
    print('  Converting punctuations...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            excCnt, queCnt, dotCnt = w.count('!'), w.count('?'), w.count('.')
            if queCnt:        s[j] = '_?'
            elif excCnt >= 5: s[j] = '_!!!'
            elif excCnt >= 3: s[j] = '_!!'
            elif excCnt >= 1: s[j] = '_!'
            elif dotCnt >= 2: s[j] = '_...'
            elif dotCnt >= 1: s[j] = '_.'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertNotWords(lines, cmap):
    print('  Converting not words...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if w[0] == '_': continue
            if w == '2':        s[j] = 'to'
            elif w.isnumeric(): s[j] = '_n'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertTailDuplicates(lines, cmap):
    print('  Converting tail duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            w = re.sub(r'(([a-z])\2{2,})$', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([a-cg-kmnp-ru-z])\2+)$', r'\g<2>', w)
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertHeadDuplicates(lines, cmap):
    print('  Converting head duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            s[j] = re.sub(r'^(([a-km-z])\2+)', r'\g<2>', w)
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertInlineDuplicates(lines, cmap, minfreq=64):
    print('  Converting inline duplicates...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            w = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([ahjkquvwxyz])\2+)', r'\g<2>', w)  # repeated 2+ times, impossible
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] > minfreq: continue
            if w == 'too': continue
            w1 = re.sub(r'(([a-z])\2+)', r'\g<2>', w) # repeated 2+ times, replace by 1
            f0, f1 = freq.get(w,0), freq.get(w1,0)
            fm = max(f0, f1)
            if fm == f0:   pass
            else:          s[j] = w1;
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertSlang(lines, cmap):
    print('  Converting slang...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if w == 'u': lines[i][j] = 'you'
            if w == 'dis': lines[i][j] = 'this'
            if w == 'dat': lines[i][j] = 'that'
            if w == 'luv': lines[i][j] = 'love'
            w1 = re.sub(r'in$', r'ing', w)
            w2 = re.sub(r'n$', r'ing', w)
            f0, f1, f2 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0)
            fm = max(f0, f1, f2)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            else:          s[j] = w2;
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertSingular(lines, cmap, minfreq=512):
    print('  Converting singular form...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
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
            cmap[original_lines[i][j]] = s[j]
    lines[i] = s

def convertRareWords(lines, cmap, min_count=16):
    print('  Converting rare words...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] < min_count: s[j] = '_r'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertCommonWords(lines, cmap):
    print('  Converting common words...')
    #beverbs = set('is was are were am s'.split())
    #articles = set('a an the'.split())
    #preps = set('to for of in at on by'.split())

    for i, s in enumerate(lines):
        #s = [word if word not in beverbs else '_b' for word in s]
        #s = [word if word not in articles else '_a' for word in s]
        #s = [word if word not in preps else '_p' for word in s]
        lines[i] = s

def convertPadding(lines, maxlen=38):
    print('  Padding...')
    for i, s in enumerate(lines):
        lines[i] = [w for w in s if w]
    for i, s in enumerate(lines):
        lines[i] = ['_', '_'] + s[:maxlen]

def preprocessLines(lines):
    global original_lines
    original_lines = lines[:]
    cmap = initializeCmap(original_lines)

    convertAccents(lines, cmap)
    convertPunctuations(lines, cmap)
    convertNotWords(lines, cmap)
    convertTailDuplicates(lines, cmap)
    convertHeadDuplicates(lines, cmap)
    convertInlineDuplicates(lines, cmap)
    convertSlang(lines, cmap)
    convertSingular(lines, cmap)
    convertRareWords(lines, cmap)
    convertCommonWords(lines, cmap)
    convertPadding(lines)
    return lines, cmap

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

def transformByConversionMap(lines, cmap, iter=2):
    for it in range(iter):
        for i, s in enumerate(lines):
            s0 = []
            for j, w in enumerate(s):
                if w in cmap:
                    if w[0] != '_':
                        s0 = s0 + cmap[w].split()
                    else:
                        s0 = s0 + [w]
            lines[i] = [w for w in s0 if w]

def transformByWord2Vec(lines, w2v):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_']

def readTestData(path):
    print('  Loading', path + '...')
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

def cmapRefine(cmap):
    cmap['yess'] = 'yes'
    cmap['pleasee'] = 'please'
    cmap['soo'] = 'so'
    cmap['noo'] = 'no'
    cmap['lovee'] = cmap['loove'] = cmap['looove'] = cmap['loooove'] = cmap['looooove'] \
        = cmap['loooooove'] = cmap['loves'] = cmap['loved'] \
        = cmap['loovee'] = cmap['lurve'] = cmap['lov'] = 'love'
    cmap['liek'] = cmap['lyk'] = cmap['lik'] = cmap['lke'] = cmap['likee'] = 'like'
    cmap['mee'] = 'me'
    cmap['hooo'] = 'hoo'
    cmap['sooon'] = cmap['soooon'] = 'soon'
    cmap['goodd'] = cmap['gud'] = 'good'
    cmap['bedd'] = 'bed'
    cmap['badd'] = 'bad'
    cmap['sadd'] = 'sad'
    cmap['madd'] = 'mad'
    cmap['redd'] = 'red'
    cmap['tiredd'] = 'tired'
    cmap['boredd'] = 'bored'
    cmap['godd'] = 'god'
    cmap['xdd'] = 'xd'
    cmap['itt'] = 'it'
    cmap['lul'] = cmap['lool'] = 'lol'
    cmap['sista'] = 'sister'
    cmap['w00t'] = 'woot'
    cmap['srsly'] = 'seriously'
    cmap['4ever'] = cmap['4eva'] = 'forever'
    cmap['neva'] = 'never'
    cmap['2day'] = 'today'
    cmap['homee'] = 'home'
    cmap['hatee'] = 'hate'
    cmap['heree'] = 'here'
    cmap['cutee'] = 'cute'
    cmap['lemme'] = 'let me'
    cmap['mrng'] = 'morning'
    cmap['gd'] = 'good'
    cmap['thx'] = cmap['thnx'] = cmap['thanx'] = cmap['thankx'] = cmap['thnk'] = 'thanks'
    cmap['ur'] = 'your'
    cmap['jaja'] = cmap['jajaja'] = cmap['jajajaja'] = 'haha'
    cmap['eff'] = cmap['ef'] = cmap['f'] = cmap['fk'] = cmap['fuk'] = cmap['fuc'] = 'fuck'
    cmap['2moro'] = cmap['2mrow'] = cmap['2morow'] = cmap['2morrow'] \
        = cmap['2morro'] = cmap['2mrw'] = cmap['2moz'] = 'tomorrow'
    cmap['babee'] = 'babe'
    cmap['theree'] = 'there'
    cmap['thee'] = 'the'
    cmap['woho'] = cmap['wohoo'] = 'woo hoo'
    cmap['2gether'] = 'together'
    cmap['2nite'] = cmap['2night'] = 'tonight'
    cmap['nite'] = 'night'
    cmap['dnt'] = 'dont'
    cmap['rly'] = 'really'
    cmap['gt'] = 'get'
    cmap['lat'] = 'late'
    cmap['dam'] = 'damn'
    cmap['4ward'] = 'forward'
    cmap['4give'] = 'forgive'
    cmap['b4'] = 'before'
    cmap['tho'] = 'though'
    cmap['kno'] = 'know'
    cmap['grl'] = 'girl'
    cmap['boi'] = 'boy'
    cmap['wrk'] = 'work'
    cmap['jst'] = 'just'
    cmap['geting'] = 'getting'
    cmap['4get'] = 'forget'
    cmap['4got'] = 'forgot'
    cmap['4real'] = 'for real'
    cmap['2go'] = 'to go'
    cmap['2b'] = 'to be'
    cmap['gr8'] = cmap['gr8t'] = cmap['gr88'] = 'great'
    cmap['str8'] = 'straight'
    cmap['twiter'] = 'twitter'
    cmap['iloveyou'] = 'i love you'
    cmap['loveyou'] = cmap['loveya'] = cmap['loveu'] = 'love you'
    cmap['xoxox'] = cmap['xox'] = cmap['xoxoxo'] = cmap['xoxoxox'] \
        = cmap['xoxoxoxo'] = cmap['xoxoxoxoxo'] = 'xoxo'
    cmap['cuz'] = cmap['bcuz'] = cmap['becuz'] = 'because'
    cmap['iz'] = 'is'
    cmap['aint'] = 'i am not'
    cmap['fav'] = 'favorite'
    cmap['pl'] = 'people'
    cmap['mah'] = 'my'
    cmap['r8'] = 'rate'
    cmap['l8'] = 'late'
    cmap['w8'] = 'wait'
    cmap['m8'] = 'mate'
    cmap['h8'] = 'hate'
    cmap['l8ter'] = cmap['l8tr'] = cmap['l8r'] = 'later'
    cmap['cnt'] = 'cant'
    cmap['fone'] = cmap['phonee'] = 'phone'
    cmap['f1'] = 'fONE'
    cmap['xboxe3'] = 'eTHREE'
    cmap['jammin'] = 'jamming'
    cmap['onee'] = 'one'
    cmap['1st'] = 'first'
    cmap['2nd'] = 'second'
    cmap['3rd'] = 'third'
    cmap['inet'] = 'internet'
    cmap['recomend'] = 'recommend'
    cmap['ah1n1'] = cmap['h1n1'] = 'hONEnONE'
    cmap['any1'] = 'anyone'
    cmap['every1'] = cmap['evry1'] = 'everyone'
    cmap['some1'] = cmap['sum1'] = 'someone'
    cmap['no1'] = 'no one'
    cmap['4u'] = 'for you'
    cmap['4me'] = 'for me'
    cmap['2u'] = 'to you'
    cmap['yu'] = 'you'
    cmap['yr'] = cmap['yrs'] = cmap['years'] = 'year'
    cmap['hr'] = cmap['hrs'] = cmap['hours'] = 'hour'
    cmap['min'] = cmap['mins'] = cmap['minutes'] = 'minute'
    cmap['go2'] = cmap['goto'] = 'go to'
    for key, value in cmap.items():
        if not key.isalpha():
            if key[-1:] == 'k':
                cmap[key] = '_n'
            if key[-2:]=='st' or key[-2:]=='nd' or key[-2:]=='rd' or key[-2:]=='th':
                cmap[key] = '_ord'
            if key[-2:]=='am' or key[-2:]=='pm' or key[-3:]=='min' or key[-4:]=='mins' \
                    or key[-2:]=='hr' or key[-3:]=='hrs' or key[-1:]=='h' \
                    or key[-4:]=='hour' or key[-5:]=='hours'\
                    or key[-2:]=='yr' or key[-3:]=='yrs'\
                    or key[-3:]=='day' or key[-4:]=='days'\
                    or key[-3:]=='wks':
                cmap[key] = '_time'

