# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders

"""
import sys

# sys.path.append('/home/rob/Dropbox/ml_projects/basket_local')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
# from util_basket import *
from util_MDN import *
from data_preprocessor import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab


class Model():
    def __init__(self, config):
        # Hyper parameters
        num_layers = config['num_layers']  # 2 layers
        hidden_size = config['hidden_size']  # hidden size 64
        max_grad_norm = config['max_grad_norm']  # 1
        batch_size = config['batch_size']  # batch size 64
        sl = config['sl']  # 39
        mixtures = config['mixtures']  # 2 mixtures
        crd = config['crd']  # 2 features
        learning_rate = config['learning_rate']
        MDN = config['MDN']
        self.sl = sl
        self.crd = crd
        self.batch_size = batch_size

        # Nodes for the input variables
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, crd, sl], name='Input_data')
        self.y_ = tf.placeholder(dtype=tf.int64, shape=[batch_size], name='Ground_truth')
        self.seq_length = tf.placeholder(tf.int32, [None], name='Sequence_length')
        self.keep_prob = tf.placeholder("float")

        # LSTM layer
        with tf.name_scope("LSTM") as scope:
            cell = tf.nn.rnn_cell.MultiRNNCell([
                lstm_cell(hidden_size, self.keep_prob) for _ in range(num_layers)
            ])

            # inputs is a list with length T=sl(39)
            # inside the inputs, the shape of each element is (batch_size, input_size)
            inputs = tf.unstack(self.x, axis=2)
            outputs, _ = tf.nn.static_rnn(cell, inputs, sequence_length=self.seq_length, dtype=tf.float32)
            # outputs, _ = tf.contrib.rnn.dynamic_rnn(cell=cell,
            #                                         inputs=inputs,
            #                                         sequence_length=seq_length,
            #                                         dtype=tf.float32)

            # the size is: (sequence_length = 39, batch_size = 64, hidden_unit = 64)
            print("The length of output: ", len(outputs))

        with tf.name_scope("SoftMax") as scope:
            final = outputs[-1]
            W_c = tf.Variable(tf.random_normal([hidden_size, 2]))
            b_c = tf.Variable(tf.constant(0.1, shape=[2]))
            self.h_c = tf.matmul(final, W_c) + b_c
            print("The shape of h_c is: ", self.h_c.shape)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.h_c, labels=self.y_)
            self.cost = tf.reduce_mean(loss)
            loss_summary = tf.summary.scalar("cross entropy_loss", self.cost)

        with tf.name_scope("Output_MDN") as scope:
            params = 6  # 5+theta
            # Two for distribution over hit&miss, params for distribution parameters
            output_units = mixtures * params
            W_o = tf.Variable(tf.random_normal([hidden_size, output_units]))
            b_o = tf.Variable(tf.constant(0.5, shape=[output_units]))
            # For comparison with XYZ, only up to last time_step
            # --> because for final time_step you cannot make a prediction
            output = outputs[:-1]
            outputs_tensor = tf.concat(output, axis=0)  # shape is [batch_size * (seq_len-1), hidden_size]
            h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)  # size: [batch_size * (seq_len-1), output_units]

        with tf.name_scope('MDN_over_next_vector') as scope:
            # Next two lines are rather ugly, But its the most efficient way to reshape the data
            h_xy = tf.reshape(h_out_tensor, (sl - 1, batch_size, output_units))
            # transpose to [batch_size, output_units, sl-1]
            h_xy = tf.transpose(h_xy, [1, 2, 0])

            # x_next = tf.slice(x,[0,0,1],[batch_size,3,sl-1])  #in size [batch_size, output_units, sl-1]
            x_next = tf.subtract(self.x[:, :2, 1:], self.x[:, :2, :sl - 1])  # offset of the coordinates?
            # From here, many variables have size [batch_size, mixtures, sl-1]
            # xn1 size: [batch_size, x, sl-1]; xn2 size: [batch_size, y, sl-1]; xn3 size: [batch_size, z, sl-1]
            xn1, xn2 = tf.split(value=x_next, num_or_size_splits=2, axis=1)
            self.mu1, self.mu2, self.s1, self.s2, self.rho, self.theta = \
                tf.split(value=h_xy, num_or_size_splits=params, axis=1)

            # make the theta mixtures
            # softmax all the theta's:
            max_theta = tf.reduce_max(self.theta, 1, keep_dims=True)
            self.theta = tf.subtract(self.theta, max_theta)
            self.theta = tf.exp(self.theta)
            normalize_theta = tf.reciprocal(tf.reduce_sum(self.theta, 1, keep_dims=True))
            self.theta = tf.multiply(normalize_theta, self.theta)

            # Deviances are non-negative and tho between -1 and 1
            self.s1 = tf.exp(self.s1)
            self.s2 = tf.exp(self.s2)
            # self.s3 = tf.exp(self.s3)
            self.rho = tf.tanh(self.rho)

            # probability in x1x2 plane
            px1x2 = tf_2d_normal(xn1, xn2, self.mu1, self.mu2,
                                 self.s1, self.s2, self.rho)
            # px3 = tf_1d_normal(xn3, self.mu3, self.s3)
            # px1x2x3 = tf.multiply(px1x2, px3)

            # Sum along the mixtures in dimension 1
            # px1x2x3_mixed = tf.reduce_sum(tf.multiply(px1x2x3, self.theta), 1)
            px1x2_mixed = tf.reduce_sum(tf.multiply(px1x2, self.theta), 1)
            print('You are using %.0f mixtures' % mixtures)
            # at the beginning, some errors are exactly zero.
            # loss_seq = -tf.log(tf.maximum(px1x2x3_mixed, 1e-20))
            loss_seq = -tf.log(tf.maximum(px1x2_mixed, 1e-20))
            self.cost_seq = tf.reduce_mean(loss_seq)
            self.cost_comb = self.cost
            if MDN:
                # The magic line where both heads come together.
                self.cost_comb += self.cost_seq

        with tf.name_scope("train") as scope:
            train_vars = tf.trainable_variables()
            # We clip the gradients to prevent explosion
            grads = tf.gradients(self.cost_comb, train_vars)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)

            # Some decay on the learning rate
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(learning_rate, global_step, 14000, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = zip(grads, train_vars)
            self.train_step = optimizer.apply_gradients(gradients, global_step=global_step)
            # The following block plots for every trainable variable
            #  - Histogram of the entries of the Tensor
            #  - Histogram of the gradient over the Tensor
            #  - Histogram of the grradient-norm over the Tensor
            self.numel = tf.constant([[0]])
            for gradient, variable in gradients:
                if isinstance(gradient, ops.IndexedSlices):
                    grad_values = gradient.values
                else:
                    grad_values = gradient

                self.numel += tf.reduce_sum(tf.size(variable))
        #
        #        h1 = tf.histogram_summary(variable.name, variable)
        #        h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
        #        h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

        with tf.name_scope("Evaluating_accuracy") as scope:
            correct_prediction = tf.equal(tf.argmax(self.h_c, 1), self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Define one op to call all summaries
        self.merged = tf.summary.merge_all()

