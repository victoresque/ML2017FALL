import sys

print(sys.argv)

method = sys.argv[1]
raw_data_path = sys.argv[2]
test_data_path = sys.argv[3]
train_feature_path = sys.argv[4]
train_label_path = sys.argv[5]
test_feature_path = sys.argv[6]
result_path = sys.argv[7]

if method == 'logistic':
    pass
elif method == 'generative':
    pass
elif method == 'best':
    pass
