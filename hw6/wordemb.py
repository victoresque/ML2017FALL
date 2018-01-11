import re
import json
import jieba
jieba.set_dictionary('data/chinese/dict.txt.big')
from gensim.models.word2vec import Word2Vec

lines = []

# 01.txt~05.txt
for i in range(5):
    with open('data/chinese/{:02d}.txt'.format(i+1), 'r', encoding='utf-8') as f:
        lines += [line[:-1] for line in f]
# 06.txt
with open('data/chinese/06.txt', 'r', encoding='utf-8') as f:
    lines += [''.join(line.split()) for line in f]
# 07.json
with open('data/chinese/07.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for data in data['data']:
        for paragraph in data['paragraphs']:
            lines += re.split(r'\n|。', paragraph['context'])

lines = [line for line in lines if line and len(line) >= 6]
lines = [[s for s in jieba.cut(line)] for line in lines]

print('Training word2vec...')
model = Word2Vec(lines, size=300, min_count=16, workers=8, iter=20)
model.save('data/word2vec')


