import tensorflow.compat.v1 as tf
import os

from .ops import linear, batch_norm, deconv2d, conv2d, lrelu


class Network(object):
    def __init__(self, scope_name):
        self._scope_name = scope_name

    def build(self, input):
        return NotImplementedError

    @property
    def all_vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self._scope_name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self._scope_name)

    def print_layers(self):
        print("Layers of {}".format(self._scope_name))
        print(self.all_vars)

    def save(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.save(sess, path)

    def load(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.restore(sess, path)


class ConditionalGenerator(Network):
    def __init__(self, output_width, output_height, output_depth, mg,
                 stride=2, kernel=4,
                 scope_name="conditionalGenerator", *args, **kwargs):
        super(ConditionalGenerator, self).__init__(scope_name=scope_name, *args, **kwargs)
        self.output_width = output_width
        self.output_height = output_height
        self.output_depth = output_depth
        self.stride = stride
        self.kernel = kernel
        self.mg = mg

    def build(self, z, y, train):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(z)[0]

            layers = [tf.concat([z, y], axis=1)]

            with tf.variable_scope("layer1"):
                layers.append(linear(layers[-1], 1024))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))


            with tf.variable_scope("layer2"):
                layers.append(linear(layers[-1], self.mg * 2 * (self.output_height // 4) * (self.output_width // 4)))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))
                layers.append(tf.reshape(layers[-1], [-1, self.output_height // 4, self.output_width // 4, self.mg * 2]))

            with tf.variable_scope("layer3"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.output_height // 2, self.output_width // 2, self.mg],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(tf.nn.relu(layers[-1]))

            with tf.variable_scope("layer4"):
                layers.append(deconv2d(
                    layers[-1],
                    [batch_size, self.output_height,
                     self.output_width, self.output_depth],
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(tf.nn.sigmoid(layers[-1]))

            return layers[-1], layers


class ACDiscriminator(Network):
    def __init__(self, num_classes, mg,
                 stride=2, kernel=4,
                 scope_name="ACDiscriminator", *args, **kwargs):
        super(ACDiscriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self._num_classes = num_classes
        self.mg = mg
        self.stride = stride
        self.kernel = kernel

    def build(self, images, train):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            layers = [images]
            with tf.variable_scope("layer1"):
                layers.append(conv2d(
                    layers[-1],
                    self.mg,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(lrelu(layers[-1]))

            with tf.variable_scope("layer2"):
                layers.append(conv2d(
                    layers[-1],
                    self.mg * 2,
                    d_h=self.stride,
                    d_w=self.stride,
                    k_h=self.kernel,
                    k_w=self.kernel))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(lrelu(layers[-1]))

            with tf.variable_scope("layer3"):
                layers.append(linear(layers[-1], 1024))
                layers.append(batch_norm()(layers[-1], train=train))
                layers.append(lrelu(layers[-1]))

            disc_layers = [layers[-1]]
            with tf.variable_scope("layer4-disc"):
                disc_layers.append(linear(disc_layers[-1], 1))

            class_layers = [layers[-1]]
            with tf.variable_scope("layer4-class"):
                class_layers.append(linear(class_layers[-1], 128))
                class_layers.append(batch_norm()(class_layers[-1], train=train))
                class_layers.append(lrelu(class_layers[-1]))
            with tf.variable_scope("layer5-class"):
                class_layers.append(linear(class_layers[-1], self._num_classes))

            layers.extend(disc_layers)
            layers.extend(class_layers)

            return disc_layers[-1], class_layers[-1], layers

