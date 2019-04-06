import pandas as pd
import numpy as np
from new_dataloader import *

# file_dir = '/Users/craneyu/PycharmProjects/basketball_predict/'
# csv_file = 'data_train_modified.csv.csv'
# data_loaded = DataLoad(file_dir, csv_file)

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

df['x_exit'] = arr_xin + arr_xout
df['y_exit'] = arr_yin + arr_yout
# save the model
df.to_csv('/Users/craneyu/PycharmProjects/basketball_predict/data_test_modified.csv')
