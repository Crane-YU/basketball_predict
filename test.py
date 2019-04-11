from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import sklearn
from tensorflow.keras import layers
from data_preprocessor import *
from model_mod import *
from util_basket import *
from util_MDN import *

# Hyper parameters
config = dict()
config['MDN'] = MDN = False  # Set to false for only the classification network
config['num_layers'] = 2  # Number of layers for the LSTM
config['hidden_size'] = 64  # Hidden size of the LSTM
config['max_grad_norm'] = 1  # Clip the gradients during training
config['batch_size'] = batch_size = 64
config['sl'] = sl = 39  # Sequence length to extract data
config['mixtures'] = 2  # Number of mixtures for the MDN
config['learning_rate'] = .005  # Initial learning rate


ratio = 0.8  # Ratio for train-val split
plot_every = 100  # How often do you want terminal output the model performances
max_iterations = 20000  # Maximum number of training iterations
dropout = 0.7  # Dropout rate in the fully connected layer
best_auc_idx = 0

file_dir = '/Users/craneyu/PycharmProjects/basketball_predict/'
plot = False  # Set True if you wish plots and visualizations

# Load the data, the name of the dataset. Note that it must end with '.csv'
csv_file = 'data_train.csv'
data_loaded = Preprocess(file_dir, csv_file)

# Munge the data. Arguments see the class
data_loaded.munge_data()
data_loaded.split_train_test(ratio=0.8)
data_dict = data_loaded.data
config['seq_length'] = data_dict['seq_len_X_train']

# X_train = data_dict['X_train']
# y_train = data_dict['y_train']
# X_val = data_dict['X_val']
# y_val = data_dict['y_val']
#
# model = tf.keras.Sequential()
# model.add(layers.Masking(mask_value=9, input_shape=(39, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.8))
#
# model.add(layers.LSTM(64, return_sequences=True))
# model.add(layers.Dropout(0.8))
# model.add(layers.BatchNormalization())
#
# model.add(layers.LSTM(64, return_sequences=False))
# model.add(layers.Dropout(0.8))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.8))
#
# model.add(layers.Dense(units=2, input_dim=64, activation='relu'))
#
# model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mae', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=32, batch_size=64, validation_data=(X_val, y_val))
# print(model.summary())

# data_loaded.export('/Users/craneyu/Desktop/', 'data_gen.csv')

X_train = np.transpose(data_dict['X_train'], [0, 2, 1])
y_train = data_dict['y_train']
X_val = np.transpose(data_dict['X_val'], [0, 2, 1])
y_val = data_dict['y_val']
N, crd, _ = X_train.shape
num_val = X_val.shape[0]

config['crd'] = crd  # Number of coordinates. usually three (X,Y,Z) and time, crd=3

# How many epochs we train
epochs = np.floor(batch_size * max_iterations / N)
print('Train with approximately %d epochs' % epochs)

model = Model(config)

# A numpy array to collect performances
perf_collect = np.zeros((7, int(np.floor(max_iterations / plot_every))))

sess = tf.Session()

# Initial settings for early stopping
auc_ma = 0.0
auc_best = 0.0

if True:
    sess.run(tf.initialize_all_variables())

    step = 0  # Step is a counter for filling the numpy array perf_collect
    i = 0
    early_stop = False
    while i < max_iterations and not early_stop:
        batch_ind = np.random.choice(N, batch_size, replace=False)
        if i % plot_every == 0:
            # Check training performance each 100 times
            if MDN:
                fetch = [model.accuracy, model.cost, model.cost_seq]
            else:
                fetch = [model.accuracy, model.cost]
            # run the sess
            result = sess.run(fetch, feed_dict={model.x: X_train[batch_ind],
                                                model.y_: y_train[batch_ind],
                                                model.seq_length: np.reshape(data_dict['seq_len_X_train'], (-1))[batch_ind],
                                                model.keep_prob: 1.0})
            perf_collect[0, step] = result[0]
            perf_collect[1, step] = cost_train = result[1]
            if MDN:
                perf_collect[4, step] = cost_train_seq = result[2]
            else:
                cost_train_seq = 0.0

            # Check validation performance
            batch_ind_val = np.random.choice(num_val, batch_size, replace=False)
            if MDN:
                fetch = [model.accuracy, model.cost, model.merged, model.h_c, model.cost_seq]
            else:
                fetch = [model.accuracy, model.cost, model.merged, model.h_c]

            result = sess.run(fetch, feed_dict={model.x: X_val[batch_ind_val],
                                                model.y_: y_val[batch_ind_val],
                                                model.seq_length: np.reshape(data_dict['seq_len_X_val'], (-1))[batch_ind_val],
                                                model.keep_prob: 1.0})
            acc = result[0]
            perf_collect[2, step] = acc
            perf_collect[3, step] = cost_val = result[1]

            if MDN:
                perf_collect[5, step] = cost_val_seq = result[4]
            else:
                cost_val_seq = 0.0

            # Perform early stopping according to AUC score on validation set
            # soft_max output
            soft_max_out = result[3]
            print("The shape of soft max output is: ", soft_max_out.shape)
            # Pick of the column in soft_max output is arbitrary.
            # If you see consistently AUC's under 0.5, then switch columns
            AUC = sklearn.metrics.roc_auc_score(y_val[batch_ind_val], soft_max_out[:, 1])
            perf_collect[6, step] = AUC
            ma_range = 5  # How many iterations to average over for AUC
            if step > ma_range:
                auc_ma = np.mean(perf_collect[6, step - ma_range + 1:step + 1])
            elif 1 < step <= ma_range:
                auc_ma = np.mean(perf_collect[6, :step + 1])

            if auc_best < AUC:
                auc_best = AUC
                best_auc_idx = i

            # # Early stop procedure
            # if auc_ma < 0.8 * auc_best:
            #     early_stop = True

            print("At %6s / %6s, validation accuracy is %5.3f, "
                  "AUC is %5.3f(%5.3f), and training loss is %5.3f / %5.3f(%5.3f)" %
                  (i, max_iterations, acc, AUC, auc_ma, cost_train, cost_train_seq, cost_val_seq))
            print("At {}, the training cost is {}, the validation cost is {}".
                  format(i, perf_collect[1, step], perf_collect[3, step]))
            print("The best AUC is %6s and the best index value is %s\n" % (auc_best, best_auc_idx))
            step += 1
        sess.run(model.train_step, feed_dict={model.x: X_train[batch_ind],
                                              model.y_: y_train[batch_ind],
                                              model.seq_length:
                                                  np.reshape(data_dict['seq_len_X_train'], (-1))[batch_ind],
                                              model.keep_prob: dropout})
        i += 1
    # In the next line we also fetch the softmax outputs
    batch_ind_val = np.random.choice(num_val, batch_size, replace=False)
    result = sess.run([model.accuracy, model.numel],
                      feed_dict={model.x: X_val[batch_ind_val], model.y_: y_val[batch_ind_val], model.keep_prob: 1.0})
    acc_test = result[0]
    print('The network has %s trainable parameters' % (result[1]))
