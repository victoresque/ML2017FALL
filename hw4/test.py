import sys
from keras.models import load_model
from preproc import *

#x_test = preprocessTestingData(sys.argv[1])
x_test = preprocessTestingData('data/testing_data.txt')

ylist = []
model_cnt = 10

for i in range(model_cnt):
    model = load_model('model/m'+str(i+1)+'.h5')
    ylist.append(model.predict(x_test, batch_size=512, verbose=True).flatten())

y = np.average(ylist, axis=0)
'''
y = np.zeros(len(ylist[0]))
for i, yi in enumerate(ylist):
    y = y + yi
y = y / model_cnt
'''
y = np.array([int(i > 0.5) for i in y])

#savePrediction(y, sys.argv[2])
savePrediction(y, 'result/prediction.csv')

