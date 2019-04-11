'''
This file contains two functions. The first one builds an LSTM RNN model,
and the second one uses the model to train the parameters and tests in test set.
Before the functions, some variables are defined. These variables can be changed during model evaluation process.
This file can be run directly on terminal line:

python train_test_LSTM.py

'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
from read_data import *
from util_MDN import *

file_dir = '/Users/craneyu/PycharmProjects/basketball_predict/'
plot = False  # Set True if you wish plots and visualizations

# Load the data, the name of the dataset. Note that it must end with '.csv'
data_loaded = Preprocess(file_dir, 'data_train.csv')
data_loaded.data_preprocess(verbose=False)
data_loaded.split_train_test(ratio=0.8)
data_dict = data_loaded.data
max_X = data_loaded.MAX_X
max_Y = data_loaded.MAX_Y
min_X = data_loaded.MIN_X
min_Y = data_loaded.MIN_Y

test_loaded = Preprocess(file_dir, 'data_test_modified.csv')
test_loaded.data_preprocess(verbose=False, max_x=max_X, max_y=max_Y, min_x=min_X, min_y=min_Y)

X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_val = data_dict['X_val']
y_val = data_dict['y_val']
X_test = test_loaded.data_list
y_test = test_loaded.labels


seq_list_train = data_dict['seq_len_X_train']
seq_list_val = data_dict['seq_len_X_val']
seq_list_test = test_loaded.seq_length

seq_len = 39  # max padding size
batch_size = 64  # training batch size
hidden_layer = 2  # number of layers of rnn cells
num_train = X_train.shape[0]
num_val = X_val.shape[0]
num_test = X_test.shape[0]

fw_n_hidden = 256
bw_n_hidden = 256

seq_l = tf.placeholder(tf.int64, name='Sequence_length')
# batch_size, max_time, features
inputs = tf.placeholder(tf.float32, [None, seq_len, 2], name='inputs')
# inputs = tf.unstack(inputs, axis=1)

targets = tf.placeholder(tf.float32, name='targets')

weight = tf.Variable(tf.random_normal([fw_n_hidden + bw_n_hidden, 2], stddev=0.01), name='weight')  # 1st choice of W_a
# weight = tf.Variable(tf.truncated_normal([fw_n_hidden, 2]), name='weight')  # 2nd choice of W_a
bias = tf.Variable(tf.constant(0.1, shape=[2]), name='bias')


'''
This function defines a RNN. 
Tf want to change to GRU, just change the:
cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
into:
cell = tf.nn.rnn_cell.GRUCell(rnn_size)
'''


def recurrent_neural_network(inputs, seq_max, w, b):

    Gstacked_fw_rnn = []
    Gstacked_bw_rnn = []
    for i in range(hidden_layer):
        Gstacked_fw_rnn.append(tf.nn.rnn_cell.GRUCell(fw_n_hidden))
        Gstacked_bw_rnn.append(tf.nn.rnn_cell.GRUCell(bw_n_hidden))

    # 建立前向和后向的三层RNN
    Gmcell_fw = tf.nn.rnn_cell.MultiRNNCell(Gstacked_fw_rnn)
    Gmcell_bw = tf.nn.rnn_cell.MultiRNNCell(Gstacked_bw_rnn)

    Gbioutputs, (out_fw_state, out_bw_state) = tf.nn.bidirectional_dynamic_rnn(Gmcell_fw,
                                                                   Gmcell_bw,
                                                                   inputs,
                                                                   sequence_length=seq_max,
                                                                   dtype=tf.float32)

    # stacked_fw_rnn = []
    # stacked_bw_rnn = []
    # for i in range(hidden_layer):
    #     stacked_fw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(fw_n_hidden))
    #     stacked_bw_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(bw_n_hidden))
    #
    # # 建立前向和后向的三层RNN
    # cell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw_rnn)
    # cell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw_rnn)
    #
    # sGbioutputs, sGoutput_state_fw, sGoutput_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
    #     [cell_fw], [cell_bw], inputs, sequence_length=seq_max, dtype=tf.float32)
    #
    # sgbresult, out_fw_state, out_bw_state = sGbioutputs, sGoutput_state_fw, sGoutput_state_bw
    # # mute this to test base on the forward state
    # outputs = tf.concat([out_fw_state[0][-1].h, out_bw_state[0][-1].h], 1)


    # get the fw state
    print(out_fw_state[-1])
    print(out_bw_state[-1])
    outputs = tf.concat([out_fw_state[-1], out_bw_state[-1]], 1)

    # num_units = [rnn_size, rnn_size]
    # cell_fw = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in num_units]
    # cell_bw = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in num_units]

    # lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(fw_n_hidden, state_is_tuple=True, forget_bias=1.0)
    # lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(bw_n_hidden, state_is_tuple=True, forget_bias=1.0)
    # outputs, fw_state, bw_state = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs,
    #                                                                       dtype=tf.float32)
    # for i, (cell_fw, cell_bw) in enumerate(zip(cell_fw, cell_bw)):
    #     pre_layer, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, pre_layer, dtype=tf.float32)
    # outputs = pre_layer

    last_output = tf.nn.dropout(outputs, 0.8)
    # prediction = tf.matmul(last_output, w) + b
    prediction = tf.nn.relu(tf.matmul(last_output, w) + b)  # size: [?, 2]

    # cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    # outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope="dynamic_rnn", sequence_length=seq_max)
    # outputs = tf.transpose(outputs, [1, 0, 2])
    # print(outputs.shape)
    # # print(outputs[1])
    # # last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="last_output")
    # last_output = outputs[-1]
    # print(last_output)
    # # print(outputs[-1])
    # prediction = tf.matmul(last_output, w) + b

    return prediction, Gbioutputs, outputs


'''
This function trains the model and tests its performance. 
After each iteration of the training, it prints out the number of iteration
and the loss of that iteration. When the training is done, prints out the trained parameters. 
After the testing, it prints out the test loss and saves the predicted values and the ground truth 
values into a new .csv file so that it is each to compare the results and evaluate the model performance. 
The file has two rows, with the first row being predicted values and second row being real values.
'''


def train_neural_network(inputs, seq_l):
    prediction, sgbresult, outputs = recurrent_neural_network(inputs, seq_l, weight, bias)

    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=targets)
    # loss_func = tf.reduce_mean(loss)
    cost = tf.losses.mean_squared_error(targets, prediction)
    loss_func = tf.reduce_mean(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 还没用以下代码: gradient clipping
    params = 6
    n_mixture = 2
    # num_output = 20 * params
    # is_training = False
    # # MDN W_a and b_a
    # output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fw_n_hidden + bw_n_hidden, num_output]))
    # output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_output]))
    # output = tf.nn.xw_plus_b(outputs, output_w, output_b)
    #
    # # size is: [batch_size * seq_length], later will be expanded to [batch_size * seq_length, 20]
    # flat_target_data = tf.reshape(self.y, [-1, 2])
    # [x1_data, x2_data] = tf.split(axis=1, num_or_size_splits=2, value=flat_target_data)
    #
    # def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    #     norm1 = tf.subtract(x1, mu1)
    #     norm2 = tf.subtract(x2, mu2)
    #     s1s2 = tf.multiply(s1, s2)
    #     z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
    #         2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
    #     negRho = 1 - tf.square(rho)
    #     result = tf.exp(tf.div(-z, 2 * negRho))
    #     denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
    #     result = tf.div(result, denom)
    #     return result
    #
    # def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
    #     result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    #     result1 = tf.multiply(result0, z_pi)
    #     result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    #     # at the beginning, some errors are exactly zero
    #     result1 = -tf.log(tf.maximum(result1, 1e-20))
    #     # result2 = tf.multiply(z_eos, eos_data) + tf.multiply(1 - z_eos, 1 - eos_data)
    #     # result2 = -tf.log(result2)
    #
    #     result = result1
    #     return tf.reduce_sum(result)
    #
    # def get_mixture_coef(output):
    #     # returns the tf slices containing mdn dist params
    #     z = output
    #     z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(axis=1, num_or_size_splits=6, value=z)
    #
    #     # process output z's into MDN paramters
    #     # softmax all the pi's:
    #     max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
    #     z_pi = tf.subtract(z_pi, max_pi)
    #     z_pi = tf.exp(z_pi)
    #     normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keep_dims=True))
    #     z_pi = tf.multiply(normalize_pi, z_pi)
    #
    #     # exponentiate the sigmas and also make corr between -1 and 1.
    #     z_sigma1 = tf.exp(z_sigma1)
    #     z_sigma2 = tf.exp(z_sigma2)
    #     z_corr = tf.tanh(z_corr)
    #
    #     return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]
    #
    # [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr] = get_mixture_coef(output)
    #
    # # self.pi = o_pi
    # # self.mu1 = o_mu1
    # # self.mu2 = o_mu2
    # # self.sigma1 = o_sigma1
    # # self.sigma2 = o_sigma2
    # # self.corr = o_corr
    #
    # loss_value = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x1_data, x2_data)
    # if is_training == True:
    #     cost = loss_value / (64 * 39)
    # else:
    #     cost = loss_value / (1 * 39)

    ####################################################################################################################

    output_units = n_mixture * params
    W_o = tf.Variable(tf.random_normal([fw_n_hidden + bw_n_hidden, output_units], stddev=0.01))
    b_o = tf.Variable(tf.constant(0.1, shape=[output_units]))


    # partitions = tf.range(64)
    # num_partitions = 64
    # data_list = tf.dynamic_partition(sgbresult, partitions, num_partitions, name='dynamic_unstack')
    #
    # # print(partitioned.shape)
    #
    # # data_list = tf.unstack(partitioned)
    #
    # for i in data_list:
    #     outputs_tensor.append(i[:-1, :])
    # outputs_tensor = tf.reshape(tf.stack(outputs_tensor), (-1, fw_n_hidden + bw_n_hidden))
    # print(outputs_tensor.shape)

########################################################################################################################
    # outputs_tensor = sgbresult[:, :-1, :]
    # outputs_tensor = tf.reshape(outputs_tensor, [-1, fw_n_hidden + bw_n_hidden])
    #
    # h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)
    # print("h_out_tensor shape is: ", h_out_tensor.shape)
    #
    # h_xy = tf.reshape(h_out_tensor, (seq_len-1, -1, output_units))
    # print("h_xy shape is: ", h_xy.shape)
    #
    # h_xy = tf.transpose(h_xy, [1, 2, 0])
    # print("h_xy shape is: ", h_xy.shape)
    #
    # MDN_X = tf.transpose(inputs, [0, 2, 1])
    # print("MDN_X shape is: ", MDN_X.shape)
    #
    # x_next = tf.subtract(MDN_X[:, :, 1:], MDN_X[:, :, :seq_len-1])
    # print("x_next shape is: ", x_next.shape)
    #
    # xn1, xn2 = tf.split(value=x_next, num_or_size_splits=2, axis=1)
    # print("xn1 shape is: ", xn1.shape)
    # print("xn2 shape is: ", xn2.shape)
    #
    # mu1, mu2, s1, s2, rho, theta = tf.split(value=h_xy, num_or_size_splits=params, axis=1)
    # print("mu1 shape is: ", mu1.shape)
    # print("mu2 shape is: ", mu2.shape)
    # print("s1 shape is: ", s1.shape)
    # print("s2 shape is: ", s2.shape)
    # print("rho shape is: ", rho.shape)
    # print("theta shape is: ", theta.shape)
    #
    # # make the theta mixtures
    # # softmax all the theta's:
    # max_theta = tf.reduce_max(theta, 1, keep_dims=True)
    # theta = tf.subtract(theta, max_theta)
    # theta = tf.exp(theta)
    # normalize_theta = tf.reciprocal(tf.reduce_sum(theta, 1, keep_dims=True))
    # theta = tf.multiply(normalize_theta, theta)
    #
    # # Deviances are non-negative and tho between -1 and 1
    # s1 = tf.exp(s1)
    # s2 = tf.exp(s2)
    # rho = tf.tanh(rho)
    #
    # px1x2 = tf_2d_normal(xn1, xn2, mu1, mu2, s1, s2, rho)
    # # Sum along the mixtures in dimension 1
    # px1x2_mixed = tf.reduce_sum(tf.multiply(px1x2, theta), 1)
    # # at the beginning, some errors are exactly zero.
    # loss_seq = -tf.log(tf.maximum(px1x2_mixed, 1e-20))
    # cost_seq = tf.reduce_mean(loss_seq)
    # total_cost = loss_func + cost_seq
    # train_var = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(total_cost, train_var), 1)
    # global_step = tf.Variable(0, trainable=False)
    # lr = tf.train.exponential_decay(0.005, global_step, 14000, 0.95, staircase=True)
    # optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(grads, train_var), global_step=global_step)

###################################################################################################

    train_var = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_func, train_var), 1)
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.0005, global_step, 10000, 0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(grads, train_var), global_step=global_step)

##################################################################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0
        train_cost_list = []
        dev_cost_list = []
        train_epoch_loss = 0
        dev_epoch_loss = 0
        while iteration < 20000:
            batch_ind = np.random.choice(num_train, batch_size, replace=False)

            sess.run(optimizer, feed_dict={inputs: X_train[batch_ind],
                                           targets: y_train[batch_ind],
                                           seq_l: [seq_list_train[i] for i in batch_ind]})

            c_train = sess.run(loss_func, feed_dict={inputs: X_train[batch_ind],
                                                      targets: y_train[batch_ind],
                                                      seq_l: [seq_list_train[i] for i in batch_ind]})
            train_epoch_loss += c_train/batch_size
            if iteration % 100 == 0 and iteration != 0:
                print('Train iteration', iteration, ', train loss:', train_epoch_loss/iteration)
                train_cost_list.append(train_epoch_loss/iteration)

            batch_ind_val = np.random.choice(num_val, batch_size, replace=False)
            c_val = sess.run(loss_func, feed_dict={inputs: X_val[batch_ind_val],
                                                    targets: y_val[batch_ind_val],
                                                    seq_l: [seq_list_val[i] for i in batch_ind_val]})
            dev_epoch_loss += c_val/batch_size
            if iteration % 100 == 0 and iteration != 0:
                print('Train iteration', iteration, ', validation loss:', dev_epoch_loss / iteration)
                dev_cost_list.append(dev_epoch_loss/iteration)

            iteration += 1

        plt.plot(range(0, len(train_cost_list)), train_cost_list)
        plt.plot(range(0, len(dev_cost_list)), dev_cost_list)
        plt.title('iteration vs. epoch cost, university')
        plt.show()

        test_prediction = list()
        for i in range(0, num_test):

            if i % 1000 == 0:
                print("Reading data at {0}th round".format(i))

            pre = sess.run(prediction, feed_dict={inputs: np.expand_dims(X_test[i], axis=0),
                                                  targets: np.expand_dims(y_test[i], axis=0),
                                                  seq_l: [seq_list_test[i]]})
            pre = np.array(pre)
            # print(pre.shape)
            # x is pre[0], y is pre[1]

            if ((3750901.5068-min_X) / (max_X-min_X) <= pre[0, 0] <= (3770901.5068-min_X) / (max_X-min_X)) and \
                    ((-19268905.6133-min_Y)/(max_Y-min_Y) <= pre[0, 1] <= (-19208905.6133-min_Y)/(max_Y-min_Y)):
                test_prediction.append(1)
            else:
                test_prediction.append(0)

        # Save predicted data and ground truth data into a .csv file.
        hash_id = test_loaded.id_list
        df = pd.DataFrame(columns=["id", "target"])
        df["id"] = hash_id
        df["target"] = test_prediction
        df.to_csv('final_result.csv')


train_neural_network(inputs, seq_l)
