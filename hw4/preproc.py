from util import *

lines, labels = readData('data/training_label.txt')
lines = [removeCommonWords(line) for line in lines]
lines, dictionary = getDictionaryAndTransform(lines)

print(len(dictionary))

maxlen = 0
for line in lines:
    maxlen = max(len(line), maxlen)

print(maxlen)
