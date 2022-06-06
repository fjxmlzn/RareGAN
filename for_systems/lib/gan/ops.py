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
