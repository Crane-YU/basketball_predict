import pandas as pd
import numpy as np
from data_preprocessor import *

# file_dir = '/Users/craneyu/PycharmProjects/basketball_predict/'
# csv_file = 'data_train_modified.csv.csv'
# data_loaded = DataLoad(file_dir, csv_file)

def cal_rank(idx_list):
    len_idx = len(idx_list)
    new_list = []
    idx = 0
    for i in range(0, len_idx):
        if i == len_idx - 1:
            new_list.append(idx + 1)
            break
        if idx_list[i] == idx_list[i + 1]:
            idx = idx + 1
            new_list.append(idx)
        else:
            new_list.append(idx + 1)
            idx = 0
        i = i + 1
    return new_list

df = pd.read_csv("data_test.csv", index_col=0).set_index('hash')
time_in = pd.to_datetime(df['time_entry'])
time_out = pd.to_datetime(df['time_exit'])
arr = ((df['x_exit'].isnull()) & (time_in == time_out)).to_numpy()
arr_xin = df['x_entry'].to_numpy()*arr
arr_xout = df['x_exit'].to_numpy()
arr_xout[np.isnan(arr_xout)] = 0
arr_yin = df["y_entry"].to_numpy()*arr
arr_yout = df["y_exit"].to_numpy()
arr_yout[np.isnan(arr_yout)] = 0
rank_list = cal_rank()

df['x_exit'] = arr_xin + arr_xout
df['y_exit'] = arr_yin + arr_yout
# save the model
df.to_csv('/Users/craneyu/PycharmProjects/basketball_predict/data_test_modified.csv')
