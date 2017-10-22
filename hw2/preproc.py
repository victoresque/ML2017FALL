import pandas as pd

data = pd.read_csv('data/train.csv')

data = data.drop('workclass', axis=1)
data = data.drop('fnlwgt', axis=1)
data = data.drop('education', axis=1)
data = data.drop('capital_loss', axis=1)

print(data)