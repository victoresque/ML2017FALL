import sys

[method,
 raw_data_path,
 test_data_path,
 train_feature_path,
 train_label_path,
 test_feature_path,
 result_path] = sys.argv[1:8]

if method == 'logistic':
    pass
elif method == 'generative':
    pass
elif method == 'best':
    pass
