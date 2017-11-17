import numpy as np
import matplotlib.pyplot as plt

f = open('../model/0.67623.log', 'r')

acc = []
val_acc = []

for line in f:
    if line[0] == '1':
        acc.append(float(line[71:77]))
        val_acc.append(float(line[108:114]))

l_tra, = plt.plot(list(np.arange(1, 201, 1)), acc, '-', label='training')
l_val, = plt.plot(list(np.arange(1, 201, 1)), val_acc, '-', label='validation')
plt.title('Training procedure')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.axis([0, 200, 0, 1])
plt.legend([l_tra, l_val], ['training', 'validation'], loc=7)
plt.show()