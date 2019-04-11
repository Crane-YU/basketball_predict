# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 15:32:09 2016

@author: rob
"""

import numpy as np
import pandas as pd
from itertools import groupby
from util_basket import *
from tensorflow.keras.preprocessing.sequence import pad_sequences


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


def get_duplicated_idx(csv_loc):
    df = pd.read_csv(csv_loc)
    time_in = pd.to_datetime(df['time_entry'])
    time_out = pd.to_datetime(df['time_exit'])
    arr = (time_in == time_out).to_numpy()
    return arr


class Preprocess():
    def __init__(self, direc, csv_file):
        """
        Init the class
        input:
        - direc: the folder with the datafiles
        - csv_file: the name of the csv file
        """
        assert direc[-1] == '/', 'Please provide a directory ending with a /'
        assert csv_file[-4:] == '.csv', 'Please provide a filename ending with .csv'
        self.csv_loc = direc + csv_file  # The location of the csv file
        self.data_list = list()  # the list where eventually will store all the data
        self.labels = list()  # the list where eventually will be all the data
        self.data = dict()  # Dictionary where the train and val date are located after split_train_test()
        self.N = 0
        self.seq_length = list()
        self.x_position = list()
        self.y_position = list()
        self.index_col = list()
        self.id_list = list()
        self.MIN_X = 0.0
        self.MIN_Y = 0.0
        self.MAX_X = 0.0
        self.MAX_Y = 0.0

    def data_preprocess(self, verbose=True, min_x=0, min_y=0, max_x=0, max_y=0):
        """
        Function to munge the data
        NOTE: this implementation assumes that the time is ticking down
        input:
        - height: the height to chop of data
        - seq_len: how long sequences you want?
        - verbose: boolean if you want to see some headers and other output
        - dist: the minimum distance to the basket
        """
        if self.data_list:
            print('You already have data in this instance. Are you calling function twice?')

        # import data
        if self.csv_loc[-14:] == "data_train.csv":
            df = pd.read_csv(self.csv_loc, index_col=0).set_index('hash')
        else:
            df = pd.read_csv(self.csv_loc, index_col=0)
            self.id_list = list(df.groupby('hash')['trajectory_id'].apply(lambda x: x.iloc[-1]).values)

        if verbose:
            print('The shape of the read data is ', df.shape)
            # # To plot a single shot
            # test = df[df['id'] == "0021500001_105"]
            # test.head(10)
        self.MIN_X = min_x
        self.MIN_Y = min_y
        self.MAX_X = max_x
        self.MAX_Y = max_y

        x_entry_arr = df['x_entry'].to_numpy(dtype=np.float64)
        y_entry_arr = df['y_entry'].to_numpy(dtype=np.float64)
        x_exit_arr = df['x_exit'].to_numpy(dtype=np.float64)
        y_exit_arr = df['y_exit'].to_numpy(dtype=np.float64)
        idx_arr = get_duplicated_idx(self.csv_loc)

        # Testing
        # print(len(idx_arr))

        for idx in range(0, x_entry_arr.shape[0]):
            if idx_arr[idx] == True:
                self.index_col.append(df.index[idx])
                self.x_position.append(x_entry_arr[idx])
                self.y_position.append(y_entry_arr[idx])
            else:
                self.index_col.append(df.index[idx])
                self.index_col.append(df.index[idx])
                self.x_position.append(x_entry_arr[idx])
                self.x_position.append(x_exit_arr[idx])
                self.y_position.append(y_entry_arr[idx])
                self.y_position.append(y_exit_arr[idx])

        # print the col length
        # print(len(self.index_col))
        rank_num = cal_rank(self.index_col)
        df1 = pd.DataFrame(columns=["x", "y", "index_col", "rank_num"])
        df1['x'] = self.x_position
        df1['y'] = self.y_position
        df1['index_col'] = self.index_col
        df1['rank_num'] = rank_num
        df_arr = df1[['x', 'y', 'rank_num']].to_numpy(dtype=np.float64)

        if self.csv_loc[-14:] == "data_train.csv":
            self.MIN_X = np.min(self.x_position)
            self.MAX_X = np.max(self.x_position)
            self.MIN_Y = np.min(self.y_position)
            self.MAX_Y = np.max(self.y_position)

        df = None
        df1 = None

        start_ind = 0
        end_ind = 0
        row, _ = df_arr.shape

        for i in range(1, row):
            if verbose and i % 1000 == 0:
                print('At line %5.0f of %5.0f' % (i, row))
            if int(df_arr[i, 2]) == 1:

                end_ind = i

                seq = df_arr[start_ind:end_ind, :2]
                if self.csv_loc[-14:] == "data_train.csv":
                    if len(seq) >= 4:
                        seq[:, 0] = (seq[:, 0] - self.MIN_X) / (self.MAX_X - self.MIN_X)
                        seq[:, 1] = (seq[:, 1] - self.MIN_Y) / (self.MAX_Y - self.MIN_Y)
                        self.data_list.append(seq[:-1, :])  # Add all the sequences to the list
                        self.labels.append(seq[-1, :])
                        self.seq_length.append(len(seq) - 1)
                else:
                    seq[:-1, 0] = (seq[:-1, 0] - self.MIN_X) / (self.MAX_X - self.MIN_X)
                    seq[:-1, 1] = (seq[:-1, 1] - self.MIN_Y) / (self.MAX_Y - self.MIN_Y)
                    self.data_list.append(seq[:-1, :])  # Add all the sequences to the list
                    if (3750901.5068 <= seq[-1, 0] <= 3770901.5068) and (-19268905.6133 <= seq[-1, 1] <= -19208905.6133):
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
                    self.seq_length.append(len(seq) - 1)

                start_ind = end_ind

        start_ind = end_ind
        seq = df_arr[start_ind:, :2]

        if self.csv_loc[-14:] == "data_train.csv":
            if len(seq) >= 4:
                seq[:, 0] = (seq[:, 0] - self.MIN_X) / (self.MAX_X - self.MIN_X)
                seq[:, 1] = (seq[:, 1] - self.MIN_Y) / (self.MAX_Y - self.MIN_Y)
                self.data_list.append(seq[:-1, :])  # Add all the sequences to the list
                self.labels.append(seq[-1, :])
                self.seq_length.append(len(seq) - 1)
        else:
            seq[:-1, 0] = (seq[:-1, 0] - self.MIN_X) / (self.MAX_X - self.MIN_X)
            seq[:-1, 1] = (seq[:-1, 1] - self.MIN_Y) / (self.MAX_Y - self.MIN_Y)
            self.data_list.append(seq[:-1, :])  # Add all the sequences to the list
            if (3750901.5068 <= seq[-1, 0] <= 3770901.5068) and (-19268905.6133 <= seq[-1, 1] <= -19208905.6133):
                self.labels.append(1)
            else:
                self.labels.append(0)
            self.seq_length.append(len(seq) - 1)

        try:
            self.data_list = pad_sequences(self.data_list, maxlen=39, value=0, dtype=np.float64, padding='post')
            self.data_list = np.stack(self.data_list, 0)
            self.labels = np.stack(self.labels, 0)
            # print("data list shape is: ", self.data_list.shape)
            # print("label shape is: ", self.labels.shape)
            self.N = len(self.labels)
        except:
            print('Something went wrong when convert list to np array')

    # def split_train_test(self, ratio=0.8):
    #     assert not isinstance(self.data_list, list), 'First munge the data before returning'
    #     N, seq_len, crd = self.data_list.shape
    #     assert seq_len > 1, 'Seq_len appears to be singleton'
    #     assert ratio < 1.0, 'Provide ratio as a float between 0 and 1'
    #     # Split the data
    #     ind_cut_train = int((ratio-0.2) * N)
    #     ind_cut_val = int(0.2*N) + ind_cut_train
    #
    #     self.data['X_train'] = self.data_list[:ind_cut_train]
    #     self.data['y_train'] = self.labels[:ind_cut_train]
    #     self.data['X_val'] = self.data_list[ind_cut_train:ind_cut_val]
    #     self.data['y_val'] = self.labels[ind_cut_train:ind_cut_val]
    #     self.data['X_test'] = self.data_list[ind_cut_val:]
    #     self.data['y_test'] = self.labels[ind_cut_val:]
    #     self.data['seq_len_X_train'] = self.seq_length[:ind_cut_train]
    #     self.data['seq_len_X_val'] = self.seq_length[ind_cut_train:ind_cut_val]
    #     self.data['seq_len_X_test'] = self.seq_length[ind_cut_val:]
    #     print('Split completed!')
    #
    #     return

    def split_train_test(self, ratio=0.8):
        assert not isinstance(self.data_list, list), 'First munge the data before returning'
        N, seq_len, crd = self.data_list.shape
        assert seq_len > 1, 'Seq_len appears to be singleton'
        assert ratio < 1.0, 'Provide ratio as a float between 0 and 1'
        # Split the data
        ind_cut = int(ratio * N)

        self.data['X_train'] = self.data_list[:ind_cut]
        self.data['y_train'] = self.labels[:ind_cut]
        self.data['X_val'] = self.data_list[ind_cut:]
        self.data['y_val'] = self.labels[ind_cut:]
        self.data['seq_len_X_train'] = self.seq_length[:ind_cut]
        self.data['seq_len_X_val'] = self.seq_length[ind_cut:]
        print('%.0f Train samples and %.0f val samples' % (ind_cut, N - ind_cut))

        return

    def export(self, folder, filename):
        """Export the data3 for use in other classifiers or programs
        input
        - folder: folder name, ending with '/'
        - filename: ending with .csv
        output
        - saves the data_windows to csv at specified context
        """
        assert folder[-1] == '/', 'Please provide a folder ending with a /'
        assert filename[-4:] == '.csv', 'Please provide a filename ending with .csv'
        data = np.reshape(self.data_list, (-1, 2))
        np.savetxt(folder + filename, data, delimiter=',')
        print('Data is saved to %s, with %.0f rows and center at %s' % (filename, data.shape[0], self.data_list))
        return

    def next_batch(batch, batch_size, filt_X, filt_Y):
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            x_batch.append(filt_X[batch * batch_size + i])
            y_batch.append(filt_Y[batch * batch_size + i])

        return x_batch, y_batch
