import tensorflow as tf
import numpy as np


def linear(input_, output_size, scope_name="linear"):
    with tf.variable_scope(scope_name):
        input_ = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        output = tf.layers.dense(
            input_,
            output_size)
        return output


def flatten(input_, scope_name="flatten"):
    with tf.variable_scope(scope_name):
        output = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        return output


class batch_norm(object):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self._epsilon = epsilon
            self._momentum = momentum
            self._name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self._momentum,
                                            updates_collections=None,
                                            epsilon=self._epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self._name)


def lrelu(x, leak=0.2, name="lrelu"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    return tf.maximum(x, leak * x)
