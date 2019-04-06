import pandas as pd
import numpy as np
from new_dataloader import *

file_dir = '/Users/craneyu/PycharmProjects/basketball_predict/'
csv_file = 'data_test_modified.csv'
# csv_loc = file_dir + csv_file
# df = pd.read_csv(csv_loc, index_col=0)
data_loaded = DataLoad(file_dir, csv_file)
data_loaded.munge_data(3)
data_loaded.split_train_test(ratio=0.8)
data_dict = data_loaded.data
# data_loaded.export('/Users/craneyu/Desktop/', 'test_gen.csv')

X_train = np.transpose(data_dict['X_train'], [0, 2, 1])
y_train = data_dict['y_train']
X_val = np.transpose(data_dict['X_val'], [0, 2, 1])
y_val = data_dict['y_val']

N, crd, _ = X_train.shape
num_val = X_val.shape[0]
