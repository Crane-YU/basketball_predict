import tensorflow as tf
import numpy as np
import random

class Model():
    def __init__(self, args):

        def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            z = tf.square((x1 - mu1) / sigma1) + tf.square((x2 - mu2) / sigma2) - \
                2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            return tf.exp(-z / (2 * (1 - tf.square(rho)))) / (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho)))

        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], dim)

        if args.action == 'train':
            args.b == 0
        self.args = args

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_len, 2])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_len, 2])

        x = tf.split(self.x, args.T, axis=1)  # T is the rnn output state
        x_list = [tf.squeeze(x_i, [1]) for x_i in x]
        if args.mode == 'predict':
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size)  # 400
            self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * args.num_layers)
            # if (args.keep_prob < 1):  # training mode
            #     self.stacked_cell = tf.nn.rnn_cell.DropoutWrapper(self.stacked_cell, output_keep_prob=args.keep_prob)
            self.init_state = self.stacked_cell.zero_state(args.batch_size, tf.float32)

            self.output_list, self.final_state = tf.nn.rnn(self.stacked_cell, x_list, self.init_state)

        num_output = 20 * 6  # 20 (# of gaussian) * (pi + 2 * (mu + sigma) + rho)

        # MDN W_a and b_a
        output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.rnn_state_size, num_output]))
        output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_output]))

        # 这句话对我的代码没有影响
        output = tf.nn.xw_plus_b(tf.reshape(tf.concat(1, self.output_list), [-1, args.rnn_state_size]),
                                      output_w, output_b)

        # size is: [batch_size * seq_length], later will be expanded to [batch_size * seq_length, 20]
        flat_target_data = tf.reshape(self.y, [-1, 2])
        [x1_data, x2_data] = tf.split(axis=1, num_or_size_splits=2, value=flat_target_data)

        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
                2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
            negRho = 1 - tf.square(rho)
            result = tf.exp(tf.div(-z, 2 * negRho))
            denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
            result = tf.div(result, denom)
            return result

        def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
            result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
            result1 = tf.multiply(result0, z_pi)
            result1 = tf.reduce_sum(result1, 1, keep_dims=True)
            # at the beginning, some errors are exactly zero
            result1 = -tf.log(tf.maximum(result1, 1e-20))
            # result2 = tf.multiply(z_eos, eos_data) + tf.multiply(1 - z_eos, 1 - eos_data)
            # result2 = -tf.log(result2)

            result = result1
            return tf.reduce_sum(result)

        # output: batch_size * rnn_state_size
        def get_mixture_coef(output):
            # returns the tf slices containing mdn dist params
            z = output
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
                axis=1, num_or_size_splits=6, value=z)

            # process output z's into MDN paramters

            # softmax all the pi's:
            max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
            z_pi = tf.subtract(z_pi, max_pi)
            z_pi = tf.exp(z_pi)
            normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keep_dims=True))
            z_pi = tf.multiply(normalize_pi, z_pi)

            # exponentiate the sigmas and also make corr between -1 and 1.
            z_sigma1 = tf.exp(z_sigma1)
            z_sigma2 = tf.exp(z_sigma2)
            z_corr = tf.tanh(z_corr)

            return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = get_mixture_coef(output)

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr

        loss_value = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x1_data, x2_data)
        #############
        self.cost = loss_value / (args.batch_size * 39)
        #############

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if accumulate >= x:
                    return i
            print('error with sampling ensemble')
            return -1

        idx = get_pi_idx(random.random(), o_pi[0])
        next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

        # no use
        self.end_of_stroke = 1 / (1 + tf.exp(self.output[:, 0]))

        # modified code: [batch_size * seq_length, 20] for each
        pi_hat, self.mu1, self.mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(self.output, 6, axis=1)

        pi_exp = tf.exp(pi_hat)
        pi_exp_sum = tf.reduce_sum(pi_exp, 1)  # size: [batch_size * seq_length] 1_D
        self.pi = pi_exp / expand(pi_exp_sum, 1, 20)  # self.pi/sum => normalized

        self.sigma1 = tf.exp(sigma1_hat)
        self.sigma2 = tf.exp(sigma2_hat)
        self.rho = tf.tanh(rho_hat)

        self.gaussian = self.pi * bivariate_gaussian(expand(y1, 1, 20), expand(y2, 1, 20), self.mu1,
                                                     self.mu2, self.sigma1, self.sigma2, self.rho)
        eps = 1e-20
        self.loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(self.gaussian, 1) + eps))

        self.loss = self.loss_gaussian / (args.batch_size * args.T)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(self.loss)

        ################################################################################################################
        ################################################################################################################

    def sample(self, sess, length):
        x = np.zeros([1, 1, 3], np.float32)
        x[0, 0, 2] = 1
        strokes = np.zeros([length, 3], dtype=np.float32)
        strokes[0, :] = x[0, 0, :]
        if self.args.mode == 'predict':
            state = sess.run(self.stacked_cell.zero_state(1, tf.float32))
        for i in range(length - 1):
            if self.args.mode == 'predict':
                feed_dict = {self.x: x, self.init_state: state}
                end_of_stroke, pi, mu1, mu2, sigma1, sigma2, rho, state = sess.run(
                    [self.end_of_stroke, self.pi, self.mu1, self.mu2,
                     self.sigma1, self.sigma2, self.rho, self.final_state],
                    feed_dict=feed_dict
                )
            x = np.zeros([1, 1, 3], np.float32)
            r = np.random.rand()
            accu = 0
            for m in range(20):
                accu += pi[0, m]
                if accu > r:
                    x[0, 0, 0:2] = np.random.multivariate_normal(
                        [mu1[0, m], mu2[0, m]], [[np.square(sigma1[0, m]), rho[0, m] * sigma1[0, m] * sigma2[0, m]],
                        [rho[0, m] * sigma1[0, m] * sigma2[0, m], np.square(sigma2[0, m])]]
                    )

                    break
            e = np.random.rand()
            if e < end_of_stroke:
                x[0, 0, 2] = 1
            else:
                x[0, 0, 2] = 0
            strokes[i + 1, :] = x[0, 0, :]
        return strokes