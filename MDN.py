import numpy as np
import tensorflow as tf


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """ 2D normal distribution
    input:
    - x, mu: input vectors
    - s1, s2: standard deviances over x1 and x2
    - rho: correlation coefficient in x1-x2 plane
    """

    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    z = tf.square(norm1 / s1) + tf.square(norm2 / s2) - 2.0 * (tf.multiply(rho, tf.multiply(norm1, norm2)) / s1s2)
    numerator = tf.exp((-1.0 * z) / (2.0 * (1 - tf.square(rho))))
    denominator = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(1 - tf.square(rho)))
    px1x2 = tf.div(numerator, denominator)
    return px1x2
