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


def return_large_true(ind_crit):
    """
    Calculate the longest consecutive True's in ind_crit
    For use in selecting indices from the sequence. ind_crit
    is a boolean array where criteria are met. This function
    return the index (best_i) and the length (best_elems) of
    consecutive True's
    """
    i = 0
    best_elems = 0
    best_i = 0

    for key, group in groupby(ind_crit, lambda x: x):
        number = next(group)
        elems = len(list(group)) + 1
        if number == 1 and elems > 1:
            if elems > best_elems:
                best_elems = elems
                best_i = i
        i += elems
    return best_elems, best_i


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


class DataLoad():
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
        self.data_list = []  # the list where eventually will be all the data
        # After munging, the data3 will be in [n, seq_len, crd]
        self.labels = []  # the list where eventually will be all the data
        self.is_abs = True  # Boolean to indicate if we have absolute data or offset data
        self.data = {}  # Dictionary where the train and val date are located after split_train_test()
        # Count the epochs
        self.N = 0
        self.iter_train = 0
        self.epochs = 0
        # self.omit = 0  # A counter for how many sequences didn't match criteria
        self.x_position = list()
        self.y_position = list()
        self.index_col = list()

    def munge_data(self, seq_len=8, verbose=True):
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

        if verbose:
            print('The shape of the read data is ', df.shape)
            # # To plot a single shot
            # test = df[df['id'] == "0021500001_105"]
            # test.head(10)

        df_arr = df.as_matrix(['x_entry', 'y_entry', 'x_exit', 'y_exit'])
        x_entry_arr = df_arr[:, 0]
        y_entry_arr = df_arr[:, 1]
        x_exit_arr = df_arr[:, 2]
        y_exit_arr = df_arr[:, 3]
        idx_arr = get_duplicated_idx(self.csv_loc)
        print(len(idx_arr))

        for idx in range(0, df_arr.shape[0]):
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
        print(len(self.index_col))
        rank_num = cal_rank(self.index_col)
        df1 = pd.DataFrame(columns=["x", "y", "index_col", "rank_num"])
        df1['x'] = self.x_position
        df1['y'] = self.y_position
        df1['index_col'] = self.index_col
        df1['rank_num'] = rank_num
        df_arr = df1.as_matrix(['x', 'y', 'rank_num', 'index_col'])
        MIN_X = np.min(self.x_position)
        MAX_X = np.max(self.x_position)
        MIN_Y = np.min(self.y_position)
        MAX_Y = np.max(self.y_position)

        # get the array shape
        # print("The dr_arr shaope is:", df_arr.shape)
        df = None
        df1 = None

        start_ind = 0
        end_ind = 0
        row, _ = df_arr.shape

        for i in range(1, row):
            if verbose and i % 1000 == 0:
                print('At line %5.0f of %5.0f' % (i, row))
            if int(df_arr[i, 2]) == 1:
                # print(df_arr[i, 3])
                end_ind = i  # Note this represent the final index + 1
                # Now we have the start index and end index
                seq = df_arr[start_ind:end_ind, :2]  # get the cols for x, y

                # 保证有4个以上的连续的True值
                # if seq.shape[0] > seq_len:
                #     self.padding_len = seq.shape[0]
                self.labels.append(seq[-1, :])

                    # if (3750901.5068 <= seq[-1, 0] <= 3770901.5068) & (-19268905.6133 <= seq[-1, 1] <= -19208905.6133):
                    #     self.labels.append(1)
                    # else:
                    #     self.labels.append(0)
                    # seq[-seq_len-1:-1, 0] = (seq[-seq_len-1:-1, 0] - MIN_X) / (MAX_X - MIN_X)
                    # seq[-seq_len-1:-1, 1] = (seq[-seq_len-1:-1, 1] - MIN_Y) / (MAX_Y - MIN_Y)
                self.data_list.append(seq[:-1, :])  # Add all the sequences to the list

                # else:
                #     self.omit += 1
                start_ind = end_ind

        start_ind = end_ind
        seq = df_arr[start_ind:, :2]
        # if seq.shape[0] > seq_len:
        #     self.padding_len = seq.shape[0]
            # if (3750901.5068 <= seq[-1, 0] <= 3770901.5068) & (-19268905.6133 <= seq[-1, 1] <= -19208905.6133):
            #     self.labels.append(1)
            # else:
            #     self.labels.append(0)
            # seq[-seq_len - 1:-1, 0] = (seq[-seq_len - 1:-1, 0] - MIN_X) / (MAX_X - MIN_X)
            # seq[-seq_len - 1:-1, 1] = (seq[-seq_len - 1:-1, 1] - MIN_Y) / (MAX_Y - MIN_Y)
        self.labels.append(seq[-1, :])
        self.data_list.append(seq[:-1, :])  # Add all the sequences to the list
        # else:
        #     self.omit += 1

        try:
            self.data_list = pad_sequences(self.data_list)
            self.data_list = np.stack(self.data_list, 0)
            self.labels = np.stack(self.labels, 0)
            print("data list shape is: ", self.data_list.shape)
            print("label shape is: ", self.labels.shape)
            self.N = len(self.labels)
        except:
            print('Something went wrong when convert list to np array')
        # print('In the munging, we lost %.0f sequences (%.2f) that did not match criteria' %
        #       (self.omit, float(self.omit) / self.N))

    # def entropy_offset(self):
    #     """Calculates the self entropy for all the offsets"""
    #     offset = self.data3[:, 1:, :3] - self.data3[:, :-1, :3]
    #     offset = np.reshape(offset, (-1, 3))
    #     cov = np.cov(offset, rowvar=False)
    #     print(cov)
    #     ent = 0.5 * np.log(np.power(2 * np.pi * np.e, 3) * np.linalg.det(cov))
    #     print(ent)
    #     return

    # def center_data(self, center_cent=np.array([5.25, 25.0, 10.0])):
    #     assert not isinstance(self.data3, list), 'First munge the data before centering'
    #     assert isinstance(center_cent, np.ndarray), 'Please provide the center as a numpy array'
    #     self.data3[:, :, :3] = self.data3[:, :, :3] - center_cent # normalization?
    #     self.center -= center_cent
    #     print('New center', self.center)
    #     return

    def abs_to_off(self):
        assert self.is_abs, 'Your data is already offset'
        assert not isinstance(self.data_list, list), 'First munge the data before returning'
        off = self.data_list[:, 1:, :3] - self.data_list[:, :-1, :3]
        time = np.expand_dims(self.data_list[:, 1:, 3], axis=2)
        self.data_list = np.concatenate((off, time), axis=2)
        self.is_abs = False
        print('Data is now offset data')
        return

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
        print('%.0f Train samples and %.0f val samples' % (ind_cut, N - ind_cut))

        return

    def sample_batch(self, inputs, result, mode='train', batch_size=64, verbose=True):
        """Samples a batch from data. Mode indicates train or val """
        assert mode in ['train', 'val'], 'Only val or train for sample_batch'
        N, sl, crd = self.data['X_' + mode].shape
        assert batch_size < N, 'Batch size exceeds total size of X_' + mode
        ind = np.random.choice(N, batch_size, replace=False)
        feed_list = []
        temp_dict = {}
        for i in range(sl):
            temp_dict.update({inputs[i]: self.data['X_' + mode][ind, i, :4]})
        labels_sub = self.data['y_' + mode][ind].astype(int)

        reverse = False
        if not reverse:
            label_one_hot = np.zeros((batch_size, 2))
            for i in range(batch_size):
                label_one_hot[i, labels_sub[i]] = 1
        else:
            label_one_hot = np.ones((batch_size, 2))
            for i in range(batch_size):
                label_one_hot[i, labels_sub[i]] = 0

        temp_dict.update({result: label_one_hot})
        return temp_dict

    def return_data_list(self, ratio=0.8, ret_list=True):
        """
        From data3 in [N,seq_len,crd] returns a list of [N,crd] with seq_len elements
        """
        assert not isinstance(self.data3, list), 'First munge the data before returning'
        N, seq_len, crd = self.data3.shape
        assert seq_len > 1, 'Seq_len appears to be singleton'
        assert ratio < 1.0, 'Provide ratio as a float between 0 and 1'

        data = {}  # dictionary for the data

        # Split the data
        # Shuffle the data
        ind_cut = int(ratio * N)
        ind = np.random.permutation(N)
        data['X_train'] = self.data3[:ind_cut]
        data['X_val'] = self.data3[ind_cut:]
        data['y_train'] = self.labels[:ind_cut]
        data['y_val'] = self.labels[ind_cut:]
        if ret_list:  # Do you want train data as list or 3D np array
            for key in ['X_train', 'X_val']:
                listdata = []
                for i in range(seq_len):
                    listdata.append(data[key][:, i, :])
                data[key] = listdata
        print('Returned data with center %s' % (self.center))

        return data

    def plot_traj_2d(self, Nplot, extra_title=' '):
        """Plots N trajectories in 2D plane. That is XY versus Z
    N: the number of trajectories within the plot"""
        fig = plt.figure()

        data2 = np.linalg.norm(self.data['X_train'][:, :, :2], axis=2)
        data2 = np.dstack((data2, self.data['X_train'][:, :, 2]))
        N = data2.shape[0]
        for i in range(Nplot):
            ind = np.random.randint(0, N)
            if self.data['y_train'][ind] == 1:
                plt.plot(data2[ind, :, 0], data2[ind, :, 1], 'r', label='miss')
            if self.data['y_train'][ind] == 0:
                plt.plot(data2[ind, :, 0], data2[ind, :, 1], 'b', label='hit')
        plt.title('Example trajectories ' + extra_title)
        plt.xlabel('Distance to basket (feet)')
        plt.ylabel('Height (feet)')
        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)
        plt.show()

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
