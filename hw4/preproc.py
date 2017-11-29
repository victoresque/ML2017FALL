from util import *
from param import *

print('Loading data...')
lines, labels = readData('data/training_label.txt')
lines2 = readData('data/training_nolabel.txt', label=False)
for s in lines2:
    lines.append(s)
    
print(len(lines))

print('Preprocessing data...')
lines = preprocessLines(lines)
dictionary = getDictionary(lines)
print('Dictionary size:', len(dictionary))

with open('data/pre_training_label.txt', 'w') as f:
    for line in lines:
        f.write(' '.join(line)+'\n')
