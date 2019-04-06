from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import sklearn
from tensorflow.keras import layers
from new_dataloader import *
from sklearn.model_selection import StratifiedKFold

# seed = 7
# np.random.seed(seed)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#
file_dir = '/Users/craneyu/PycharmProjects/basketball_predict/'
plot = False  # Set True if you wish plots and visualizations

# Load the data, the name of the dataset. Note that it must end with '.csv'
csv_file = 'data_train.csv'
data_loaded = DataLoad(file_dir, csv_file)

# Munge the data. Arguments see the class
data_loaded.munge_data(7)
data_loaded.split_train_test(ratio=0.8)
data_dict = data_loaded.data

X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_val = data_dict['X_val']
y_val = data_dict['y_val']

model = tf.keras.Sequential()
model.add(layers.Masking(mask_value=0, input_shape=(39, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.8))

model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.8))
model.add(layers.BatchNormalization())

model.add(layers.LSTM(64, return_sequences=False))
model.add(layers.Dropout(0.8))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.8))

model.add(layers.Dense(units=2, input_dim=64, activation='relu'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=32, batch_size=64, validation_data=(X_val, y_val))
print(model.summary())

# data_loaded.export('/Users/craneyu/Desktop/', 'data_gen.csv')

# X_train = np.transpose(data_dict['X_train'], [0, 2, 1])
# y_train = data_dict['y_train']
# X_val = np.transpose(data_dict['X_val'], [0, 2, 1])
# y_val = data_dict['y_val']
# cvscores = []
# N, crd, _ = X_train.shape
# num_val = X_val.shape[0]

# model = tf.keras.Sequential()
# # Adds a densely-connected layer with 64 units to the model:
# model.add(layers.Dense(128, input_shape=(2, 8), activation='relu'))
# # Add another:
# model.add(layers.Dense(64, activation='relu'))
# # Add a softmax layer with 10 output units:
# model.add(layers.Flatten())
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='mse',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train.reshape(-1, 1), epochs=10, batch_size=64)
# model.evaluate(X_val, y_val, batch_size=32)

# for train, test in kfold.split(X_train, y_train):
#     model = tf.keras.Sequential()
#     # Adds a densely-connected layer with 64 units to the model:
#     model.add(layers.Dense(128, input_shape=(2, 8), activation='relu'))
#     # Add another:
#     model.add(layers.Dense(64, activation='relu'))
#     # Add a softmax layer with 10 output units:
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#                   loss='mse',
#                   metrics=['accuracy'])
#
#     model.fit(X_train[train], y_train[train].reshape(-1, 1), epochs=10, batch_size=64, verbose=0)
#     scores = model.evaluate(X_train[test], y_train[test], batch_size=32)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#     cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
