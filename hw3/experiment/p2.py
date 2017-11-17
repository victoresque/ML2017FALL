import numpy as np
import matplotlib.pyplot as plt

f = open('dnn.log', 'r')

acc = []
val_acc = []

for line in f:
    if line[0:5] == '25121':
        acc.append(float(line[72:78]))
        val_acc.append(float(line[109:115]))

plt.plot(list(np.arange(1, 201, 1)), acc, '-', val_acc, '-')
plt.title('Training procedure')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.axis([0, 200, 0, 1])
plt.show()