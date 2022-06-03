import tensorflow as tf
import numpy as np
if float('.'.join(tf.__version__.split('.')[:2])) >= 2:
    tf = tf.compat.v1
    import tf_slim as slim
    batch_norm_fn = slim.batch_norm
else:
    batch_norm_fn = tf.contrib.layers.batch_norm


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
        return batch_norm_fn(x,
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


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(
            'w',
            [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable(
            'biases',
            [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        return deconv


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w',
            [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            input_,
            w,
            strides=[1, d_h, d_w, 1],
            padding='SAME')

        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(
            tf.nn.bias_add(conv, biases),
            [-1] + conv.get_shape().as_list()[1:])

        return conv
