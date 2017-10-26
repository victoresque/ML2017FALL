from sklearn.externals import joblib
from util import *

clf = joblib.load('model/model.pkl')

X_test = load_test_data('data/X_test.csv')
y_test = clf.predict(X_test).reshape((-1, 1))

save_prediction(y_test, 'result/prediction.csv')