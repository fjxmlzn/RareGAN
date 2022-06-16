import tensorflow.compat.v1 as tf
import os

from .ops import linear, batch_norm, lrelu
from lib.data.fields import BitField, ChoiceField, FixedField


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
    def __init__(self,
                 num_layers, l_dim, input_definition,
                 scope_name="conditionalGenerator", *args, **kwargs):
        super(ConditionalGenerator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self._num_layers = num_layers
        self._l_dim = l_dim
        self._input_definition = input_definition

    def build(self, z, condition, train):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            layers = [z]

            for i in range(self._num_layers - 1):
                with tf.variable_scope("layer{}".format(i)):
                    layers.append(tf.concat([layers[-1], condition], axis=1))
                    layers.append(linear(layers[-1], self._l_dim))
                    layers.append(batch_norm()(layers[-1], train=train))
                    layers.append(tf.nn.relu(layers[-1]))

            class_layers = []
            softmax_layers = []
            samples = []
            probs = []

            with tf.variable_scope("layer{}".format(self._num_layers - 1)):
                layers.append(tf.concat([layers[-1], condition], axis=1))

                for field_i, field in enumerate(self._input_definition):
                    with tf.variable_scope("field{}".format(field_i)):
                        if isinstance(field, (BitField, ChoiceField)):
                            if isinstance(field, BitField):
                                num_classes = 2
                            else:
                                num_classes = len(field.choices)
                            class_layers.append(linear(
                                layers[-1], num_classes * field.numpy_dim))
                            class_layers.append(tf.reshape(
                                class_layers[-1],
                                [-1, field.numpy_dim, num_classes]))
                            softmax = tf.nn.softmax(class_layers[-1], axis=2)
                            softmax_layers.append(tf.reshape(
                                softmax,
                                [-1, num_classes * field.numpy_dim]))
                            sample = tf.argmax(class_layers[-1], axis=2)
                            prob = tf.reduce_max(softmax, axis=2)

                            samples.append(sample)
                            probs.append(prob)
                        elif isinstance(field, FixedField):
                            print("Skipping field {}".format(field.name))
                        else:
                            raise Exception("Unknown field {}".format(field))

            layers.extend(class_layers)
            layers.extend(softmax_layers)

            samples = tf.concat(samples, 1)
            probs = tf.concat(probs, 1)
            outputs = tf.concat(softmax_layers, 1)

            return samples, probs, outputs, layers


class ACDiscriminator(Network):
    def __init__(self,
                 num_shared_layers, num_disc_layers, num_class_layers,
                 l_dim, num_classes,
                 scope_name="ACDiscriminator", *args, **kwargs):
        super(ACDiscriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self._num_shared_layers = num_shared_layers
        self._num_disc_layers = num_disc_layers
        self._num_class_layers = num_class_layers
        self._l_dim = l_dim
        self._num_classes = num_classes

    def build(self, samples, train):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            layers = [samples]

            with tf.variable_scope("shared"):
                for i in range(self._num_shared_layers):
                    with tf.variable_scope("layer{}".format(i)):
                        layers.append(linear(layers[-1], self._l_dim))
                        layers.append(lrelu(layers[-1]))

            disc_layers = [layers[-1]]
            with tf.variable_scope("disc"):
                for i in range(self._num_disc_layers - 1):
                    with tf.variable_scope("layer{}".format(i)):
                        disc_layers.append(linear(
                            disc_layers[-1], self._l_dim))
                        disc_layers.append(lrelu(disc_layers[-1]))
                with tf.variable_scope("layer{}".format(
                        self._num_disc_layers - 1)):
                    disc = linear(disc_layers[-1], 1)
                    disc_layers.append(disc)

            class_layers = [layers[-1]]
            with tf.variable_scope("class"):
                for i in range(self._num_class_layers - 1):
                    with tf.variable_scope("layer{}".format(i)):
                        class_layers.append(linear(
                            class_layers[-1], self._l_dim))
                        class_layers.append(lrelu(class_layers[-1]))

                with tf.variable_scope("layer{}".format(
                        self._num_class_layers - 1)):
                    class_logits = linear(class_layers[-1], self._num_classes)
                    class_layers.append(class_logits)

            layers.extend(disc_layers)
            layers.extend(class_layers)

            return disc, class_logits, layers
