import tensorflow.compat.v1 as tf
import numpy as np
import os
import math
from tqdm import tqdm
import datetime
import sys
import scipy.special

from lib.data.fields import BitField, ChoiceField, FixedField
from .amplification_utils import amplification_to_class_labels


class RareGAN(object):
    def __init__(self, sess, blackbox, dataset, input_definition,
                 request_budget, target_threshold,
                 generator, discriminator, batch_size, z_dim,
                 balanced_disc_weights, class_loss_with_fake,
                 balanced_class_weights, high_fraction_multiple,
                 initial_random_budget, budget_per_step,
                 oversampling_ratio,
                 num_iterations_per_step,
                 checkpoint_dir, time_path,
                 extra_iteration_checkpoint_freq,
                 iteration_log_freq,
                 gen_lr=1e-3, gen_beta1=0.5,
                 disc_lr=1e-3, disc_beta1=0.5, disc_gp_coe=10.0,
                 disc_disc_coe=1.0, gen_disc_coe=1.0):
        self._sess = sess
        self._blackbox = blackbox
        self._dataset = dataset
        self._input_definition = input_definition
        self._request_budget = request_budget
        self._target_threshold = target_threshold
        self._generator = generator
        self._discriminator = discriminator
        self._batch_size = batch_size
        self._z_dim = z_dim
        self._balanced_disc_weights = balanced_disc_weights
        self._class_loss_with_fake = class_loss_with_fake
        self._balanced_class_weights = balanced_class_weights
        self._high_fraction_multiple = high_fraction_multiple
        self._initial_random_budget = initial_random_budget
        self._budget_per_step = budget_per_step
        self._oversampling_ratio = oversampling_ratio
        self._num_iterations_per_step = num_iterations_per_step
        self._checkpoint_dir = checkpoint_dir
        self._time_path = time_path
        self._extra_iteration_checkpoint_freq = extra_iteration_checkpoint_freq
        self._iteration_log_freq = iteration_log_freq
        self._gen_lr = gen_lr
        self._gen_beta1 = gen_beta1
        self._disc_lr = disc_lr
        self._disc_beta1 = disc_beta1
        self._disc_gp_coe = disc_gp_coe
        self._disc_disc_coe = disc_disc_coe
        self._gen_disc_coe = gen_disc_coe

        self._num_input_choices = []
        for field in self._input_definition:
            if isinstance(field, BitField):
                self._num_input_choices.extend([2] * field.num_bits)
            elif isinstance(field, ChoiceField):
                self._num_input_choices.extend(
                    [len(field.choices)] * field.numpy_dim)
            elif isinstance(field, FixedField):
                print('Skipping field {}'.format(field.name))
            else:
                raise Exception('Unknown field {}'.format(field))
        self._num_input_sizes = len(self._num_input_choices)

        self._num_amplification_bins = 2

        self._EPS = 1e-8
        self._MODEL_NAME = 'model'
        self._TRAINING_STATE_FILE_NAME = 'training_state.npz'

    def build(self):
        self._build_connection()
        self._build_loss()
        self._build_summary()
        self._saver = tf.train.Saver()

    def _build_connection(self):
        self._z_pl = tf.placeholder(
            tf.float32,
            [None, self._z_dim],
            name='z')
        self._z_for_class_pl = tf.placeholder(
            tf.float32,
            [None, self._z_dim],
            name='z_for_class')
        self._real_indices_pl = tf.placeholder(
            tf.int64,
            [None, self._num_input_sizes],
            name='real_id')
        self._real_indices_for_class_pl = tf.placeholder(
            tf.int64,
            [None, self._num_input_sizes],
            name='real_id_for_class')
        self._real_conditions_for_class_pl = tf.placeholder(
            tf.float32,
            [None, self._num_amplification_bins],
            name='real_condition_for_class')
        self._generated_conditions_pl = tf.placeholder(
            tf.float32,
            [None, self._num_amplification_bins],
            name='generated_condition')
        self._generated_conditions_for_class_pl = tf.placeholder(
            tf.float32,
            [None, self._num_amplification_bins],
            name='generated_condition_for_class')
        self._disc_class_weights_pl = tf.placeholder(
            tf.float32,
            [self._num_amplification_bins],
            name='disc_class_weights')
        self._class_class_weights_pl = tf.placeholder(
            tf.float32,
            [self._num_amplification_bins],
            name='class_class_weights')

        self._real_onehots = []
        for i in range(self._num_input_sizes):
            self._real_onehots.append(tf.one_hot(
                self._real_indices_pl[:, i], self._num_input_choices[i]))
        self._real_onehots = tf.concat(self._real_onehots, 1)

        self._real_onehots_for_class = []
        for i in range(self._num_input_sizes):
            self._real_onehots_for_class.append(tf.one_hot(
                self._real_indices_for_class_pl[:, i],
                self._num_input_choices[i]))
        self._real_onehots_for_class = tf.concat(
            self._real_onehots_for_class, 1)

        self._gen_indices, self._gen_probs, self._gen_outputs, _ = \
            self._generator.build(
                self._z_pl, self._generated_conditions_pl, train=True)

        (self._gen_indices_for_class, self._gen_probs_for_class,
         self._gen_outputs_for_class, _) = \
            self._generator.build(
                self._z_for_class_pl,
                self._generated_conditions_for_class_pl,
                train=True)

        (self._test_gen_indices, self._test_gen_probs,
         self._test_gen_outputs, _) = \
            self._generator.build(
                self._z_pl, self._generated_conditions_pl, train=False)

        self._disc_fake, self._disc_fake_class_logits, _ = \
            self._discriminator.build(self._gen_outputs, train=True)
        _, self._class_logits_fake, _ = self._discriminator.build(
            self._gen_outputs_for_class, train=True)

        self._disc_real, self._disc_real_class_logits, _ = \
            self._discriminator.build(self._real_onehots, train=True)
        _, self._class_logits_real, _ = self._discriminator.build(
            self._real_onehots_for_class, train=True)

        self._test_disc, self._test_class_logits, _ = \
            self._discriminator.build(self._real_onehots, train=False)

        self._generator.print_layers()
        self._discriminator.print_layers()

    def _build_loss(self):
        batch_size = tf.shape(self._real_indices_pl)[0]

        # Classification weights.
        if self._balanced_class_weights:
            class_id_fake = tf.argmax(
                self._generated_conditions_for_class_pl, axis=1)
            class_id_real = tf.argmax(
                self._real_conditions_for_class_pl, axis=1)
            self._class_fake_weights = tf.gather(
                self._class_class_weights_pl, class_id_fake)
            self._class_real_weights = tf.gather(
                self._class_class_weights_pl, class_id_real)
        else:
            self._class_fake_weights = tf.ones(shape=[batch_size])
            self._class_real_weights = tf.ones(shape=[batch_size])

        self._class_fake_weights_mean = tf.reduce_mean(
            self._class_fake_weights)
        self._class_real_weights_mean = tf.reduce_mean(
            self._class_real_weights)

        # Classification.
        self._class_loss_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self._class_logits_real,
                labels=self._real_conditions_for_class_pl) *
            self._class_real_weights)
        self._class_loss_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self._class_logits_fake,
                labels=self._generated_conditions_for_class_pl) *
            self._class_fake_weights)

        # Discriminator weights.
        if self._balanced_disc_weights:
            disc_fake_class_id = tf.argmax(
                self._disc_fake_class_logits, axis=1)
            disc_real_class_id = tf.argmax(
                self._disc_real_class_logits, axis=1)
            self._disc_fake_weights = tf.gather(
                self._disc_class_weights_pl, disc_fake_class_id)
            self._disc_real_weights = tf.gather(
                self._disc_class_weights_pl, disc_real_class_id)
        else:
            self._disc_fake_weights = tf.ones(shape=[batch_size])
            self._disc_real_weights = tf.ones(shape=[batch_size])

        self._disc_fake_weights_mean = tf.reduce_mean(self._disc_fake_weights)
        self._disc_real_weights_mean = tf.reduce_mean(self._disc_real_weights)

        # Discriminator losses.
        self._disc_loss_class_real = self._class_loss_real
        if self._class_loss_with_fake:
            self._disc_loss_class_fake = self._class_loss_fake
        else:
            self._disc_loss_class_fake = 0.0
        self._disc_loss_real = -tf.reduce_mean(
            tf.squeeze(self._disc_real, axis=1) * self._disc_real_weights)
        self._disc_loss_fake = tf.reduce_mean(
            tf.squeeze(self._disc_fake, axis=1) * self._disc_fake_weights)

        alpha = tf.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.)
        differences_packet = self._gen_outputs - self._real_onehots
        interpolates_packet = self._real_onehots + (alpha * differences_packet)

        gradients = tf.gradients(
            self._discriminator.build(
                interpolates_packet, train=True)[0],
            [interpolates_packet])
        self._slope = tf.reduce_sum(tf.square(gradients[0]),
                                    reduction_indices=[1])
        slope = tf.sqrt(self._slope + self._EPS)
        self._disc_loss_gradient_penalty = tf.reduce_mean((slope - 1.)**2)

        self._disc_loss_fake_normalized = (
            self._disc_loss_fake / self._disc_fake_weights_mean)
        self._disc_loss_real_normalized = (
            self._disc_loss_real / self._disc_real_weights_mean)
        self._disc_loss_class_fake_normalized = (
            self._disc_loss_class_fake / self._class_fake_weights_mean)
        self._disc_loss_class_real_normalized = (
            self._disc_loss_class_real / self._class_real_weights_mean)

        self._disc_loss = ((self._disc_loss_fake_normalized *
                            self._disc_disc_coe) +
                           (self._disc_loss_real_normalized *
                            self._disc_disc_coe) +
                           self._disc_loss_class_fake_normalized +
                           self._disc_loss_class_real_normalized +
                           (self._disc_loss_gradient_penalty *
                            self._disc_gp_coe))

        # Generator losses.
        self._gen_loss_fake = -tf.reduce_mean(
            tf.squeeze(self._disc_fake, axis=1) * self._disc_fake_weights)
        self._gen_loss_class_fake = self._class_loss_fake

        self._gen_loss_fake_normalized = (
            self._gen_loss_fake / self._disc_fake_weights_mean)
        self._gen_loss_class_fake_normalized = (
            self._gen_loss_class_fake / self._class_fake_weights_mean)

        self._gen_loss = (self._gen_loss_fake_normalized * self._gen_disc_coe +
                          self._gen_loss_class_fake_normalized)

        # Optimizers.
        self._disc_op = tf.train.AdamOptimizer(self._disc_lr).minimize(
            self._disc_loss, var_list=self._discriminator.trainable_vars)
        self._gen_op = tf.train.AdamOptimizer(self._gen_lr).minimize(
            self._gen_loss, var_list=self._generator.trainable_vars)

    def _build_summary(self):
        self._gen_summary = []
        self._gen_summary.append(tf.summary.scalar(
            'loss/gen/fake', self._gen_loss_fake))
        self._gen_summary.append(tf.summary.scalar(
            'loss/gen/fake_normalized', self._gen_loss_fake_normalized))
        self._gen_summary.append(tf.summary.scalar(
            'loss/gen/class_fake', self._gen_loss_class_fake))
        self._gen_summary.append(tf.summary.scalar(
            'loss/gen/class_fake_normalized',
            self._gen_loss_class_fake_normalized))
        self._gen_summary.append(tf.summary.scalar(
            'loss/gen', self._gen_loss))
        self._gen_summary.append(tf.summary.scalar(
            'gen/prob', tf.reduce_mean(self._gen_probs)))
        self._gen_summary = tf.summary.merge(self._gen_summary)

        self._disc_summary = []
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/fake', self._disc_loss_fake))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/fake_normalized', self._disc_loss_fake_normalized))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/real', self._disc_loss_real))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/real_normalized', self._disc_loss_real_normalized))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/class_fake', self._disc_loss_class_fake))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/class_fake_normalized',
            self._disc_loss_class_fake_normalized))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/class_real', self._disc_loss_class_real))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/class_real_normalized',
            self._disc_loss_class_real_normalized))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc/gp', self._disc_loss_gradient_penalty))
        self._disc_summary.append(tf.summary.scalar(
            'loss/disc', self._disc_loss))
        self._disc_summary.append(tf.summary.scalar(
            'disc/fake', tf.reduce_mean(self._disc_fake)))
        self._disc_summary.append(tf.summary.scalar(
            'disc/real', tf.reduce_mean(self._disc_real)))
        self._disc_summary.append(tf.summary.scalar(
            'disc/fake_weights', self._disc_fake_weights_mean))
        self._disc_summary.append(tf.summary.scalar(
            'disc/real_weights', self._disc_real_weights_mean))
        self._disc_summary.append(tf.summary.scalar(
            'disc/slope', tf.reduce_mean(self._slope)))
        self._disc_summary.append(tf.summary.scalar(
            'class/fake_weights', self._class_fake_weights_mean))
        self._disc_summary.append(tf.summary.scalar(
            'class/real_weights', self._class_real_weights_mean))
        self._disc_summary = tf.summary.merge(self._disc_summary)

    def _save(self, iteration_id=None, step_id=None, saver=None,
              checkpoint_dir=None, with_model=True, with_states=True,
              with_datasets=True):
        if checkpoint_dir is None:
            checkpoint_dir = self._checkpoint_dir
        if with_model:
            if iteration_id is None:
                raise ValueError('iteration_id must be provided')
            if saver is None:
                saver = self._saver
            saver.save(
                self._sess,
                os.path.join(checkpoint_dir, self._MODEL_NAME),
                global_step=iteration_id)

        if with_states:
            if step_id is None:
                raise ValueError('step_id must be provided')
            np.savez(
                os.path.join(checkpoint_dir, self._TRAINING_STATE_FILE_NAME),
                request_usage=self._request_usage,
                step_id=step_id)

        if with_datasets:
            self._dataset.dump_to_folder(checkpoint_dir)

    def load(self, checkpoint_dir=None,
             with_model=True, with_states=True, with_datasets=True):
        if checkpoint_dir is None:
            checkpoint_dir = self._checkpoint_dir
        if with_model:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # In cases where people move the checkpoint directory to another
            # place, model path indicated by get_checkpoint_state will be
            # wrong. So we get the model name and then recontruct path using
            # checkpoint_dir.
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self._saver.restore(
                self._sess, os.path.join(checkpoint_dir, ckpt_name))

        if with_states:
            states = np.load(
                os.path.join(checkpoint_dir, self._TRAINING_STATE_FILE_NAME))
            self._request_usage = states['request_usage']
            step_id = states['step_id']
        else:
            step_id = None

        if with_datasets:
            self._dataset.load_from_folder(checkpoint_dir)

        return step_id

    def disc_from(self, numpy_inputs):
        disc = []
        class_logits = []
        for i in range(int(math.ceil(
                float(numpy_inputs.shape[0]) / self._batch_size))):
            sub_disc, sub_class_logits = self._sess.run(
                [self._test_disc, self._test_class_logits],
                feed_dict={
                    self._real_indices_pl:
                        numpy_inputs[i * self._batch_size:
                                     (i + 1) * self._batch_size]})
            disc.append(sub_disc)
            class_logits.append(sub_class_logits)
        return np.vstack(disc), np.vstack(class_logits)

    def sample_from(self, z, condition):
        indices = []
        probs = []
        outputs = []
        for i in range(int(math.ceil(float(z.shape[0]) / self._batch_size))):
            sub_indices, sub_probs, sub_outputs = self._sess.run(
                [self._test_gen_indices, self._test_gen_probs,
                 self._test_gen_outputs],
                feed_dict={
                    self._z_pl:
                        z[i * self._batch_size:
                          (i + 1) * self._batch_size],
                    self._generated_conditions_pl:
                        condition[i * self._batch_size:
                                  (i + 1) * self._batch_size]})
            indices.append(sub_indices)
            probs.append(sub_probs)
            outputs.append(sub_outputs)
        return np.vstack(indices), np.vstack(probs), np.vstack(outputs)

    def sample(self, num_samples, condition=None):
        if condition is not None:
            if isinstance(condition, int):
                full_condition = [0] * self._num_amplification_bins
                full_condition[condition] = 1
                full_condition = np.tile(full_condition, [num_samples, 1])
                condition = full_condition
            return self.sample_from(
                self.gen_z(num_samples),
                condition)
        else:
            return self.sample_from(
                self.gen_z(num_samples),
                self.gen_conditions(num_samples))

    def gen_z(self, batch_size):
        return np.random.uniform(-1.0, 1.0, [batch_size, self._z_dim])

    def gen_conditions(self, batch_size):
        _, amplifications = self._dataset.uniformly_sample(
            batch_size, condition_filter=-1)
        conditions = amplification_to_class_labels(
            amplifications, [self._target_threshold])
        return conditions

    def _add_random_data(self, num_samples, step):
        field_dict_inputs = self._input_definition.uniformly_sample_field_dict(
            num_samples)
        amplifications = self._blackbox.query(field_dict_inputs)
        numpy_inputs = self._input_definition.field_dict_to_numpy(
            field_dict_inputs)
        self._dataset.add_data(
            numpy_inputs=numpy_inputs,
            amplifications=amplifications,
            steps=step * np.ones(num_samples, dtype=np.int32),
            conditions=-np.ones(num_samples, dtype=np.int32))
        self._request_usage += num_samples

    def _add_data(self, num_samples, step):
        num_random_packets_to_test = num_samples * self._oversampling_ratio
        field_dict_inputs = self._input_definition.uniformly_sample_field_dict(
            num_random_packets_to_test)
        numpy_inputs = self._input_definition.field_dict_to_numpy(
            field_dict_inputs)
        _, class_logits = self.disc_from(numpy_inputs)
        softmax = scipy.special.softmax(class_logits, axis=1)
        max_ = np.max(softmax, axis=1)
        id_ = np.argsort(max_)
        print("Picking {} samples from {} random samples".format(
            num_samples, num_random_packets_to_test))
        print("The softmax of ranked {}th-sample is: {}".format(
            num_samples, softmax[id_[num_samples - 1]]))
        selected_numpy_inputs = numpy_inputs[id_[:num_samples]]
        selected_field_dict_inputs = \
            self._input_definition.numpy_to_field_dict(selected_numpy_inputs)
        amplifications = self._blackbox.query(selected_field_dict_inputs)
        num_selected_samples = selected_numpy_inputs.shape[0]
        self._dataset.add_data(
            numpy_inputs=selected_numpy_inputs,
            amplifications=amplifications,
            steps=step * np.ones(num_selected_samples, dtype=np.int32),
            conditions=-2 * np.ones(num_selected_samples, dtype=np.int32))
        self._request_usage += num_selected_samples

        sys.stdout.flush()

    def _update_class_weights(self):
        self._dataset.thresholds = [self._target_threshold]
        print('Effective threshold: {}'.format(self._target_threshold))


        high_fraction = self._dataset.get_amplification_fraction(
            amplification=self._target_threshold,
            condition_filter=-1)
        self._training_disc_class_weights = \
            self._dataset.compute_class_weights(
                condition_filter=-1,
                last_fraction=high_fraction * self._high_fraction_multiple)
        if self._training_disc_class_weights[0] > 1:
            self._training_disc_class_weights = np.asarray([1.0, 1.0])
        print('Disc class weights: {}'.format(
            self._training_disc_class_weights))

        self._training_class_class_weights = \
            self._dataset.compute_class_weights(
                last_fraction=high_fraction * self._high_fraction_multiple)
        if self._training_class_class_weights[0] > 1:
            self._training_class_class_weights = np.asarray([1.0, 1.0])
        print('Class class weights: {}'.format(
            self._training_class_class_weights))

        sys.stdout.flush()

    def _train_one_iteration(self, summary_writer, iteration_id):
        batch_z = self.gen_z(self._batch_size)
        batch_z_for_class = self.gen_z(self._batch_size)

        batch_real_numpy_inputs = \
            self._input_definition.uniformly_sample_numpy(
                num=self._batch_size, progress_bar=False)

        batch_real_numpy_inputs_for_class, amplifications = \
            self._dataset.uniformly_sample(self._batch_size)
        batch_real_conditions_for_class = \
            amplification_to_class_labels(
                amplifications, [self._target_threshold])

        batch_generated_conditions = self.gen_conditions(
            self._batch_size)

        feed_dict = {
            self._z_pl: batch_z,
            self._z_for_class_pl: batch_z_for_class,
            self._real_indices_pl: batch_real_numpy_inputs,
            self._real_indices_for_class_pl:
                batch_real_numpy_inputs_for_class,
            self._real_conditions_for_class_pl:
                batch_real_conditions_for_class,
            self._generated_conditions_pl:
                batch_generated_conditions,
            self._generated_conditions_for_class_pl:
                batch_real_conditions_for_class,
            self._disc_class_weights_pl:
                self._training_disc_class_weights,
            self._class_class_weights_pl:
                self._training_class_class_weights
        }

        summary_result, _ = self._sess.run(
            [self._disc_summary, self._disc_op],
            feed_dict=feed_dict)
        summary_writer.add_summary(
            summary_result, iteration_id)

        summary_result, _ = self._sess.run(
            [self._gen_summary, self._gen_op],
            feed_dict=feed_dict)
        summary_writer.add_summary(
            summary_result, iteration_id)

    def train(self, restore=False):
        if restore is True:
            restore_step_id = self.load(
                with_model=True,
                with_states=True,
                with_datasets=True)
            print('Loaded from step_id {}'.format(restore_step_id))
        else:
            restore_step_id = -1
            self._request_usage = 0
            tf.global_variables_initializer().run()

        saver = tf.train.Saver()

        step_id = restore_step_id + 1
        while True:
            with open(self._time_path, 'a') as f:
                time = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S.%f')
                f.write('step {} starts: {}\n'.format(step_id, time))

            # Add initial training data.
            if step_id == 0:
                if self._initial_random_budget > 0:
                    self._add_random_data(
                        num_samples=self._initial_random_budget,
                        step=step_id)
            else:
                budget = (self._initial_random_budget +
                          step_id * self._budget_per_step -
                          self._request_usage)
                self._add_data(
                    num_samples=budget,
                    step=step_id)

            with open(self._time_path, 'a') as f:
                time = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S.%f')
                f.write('step {} starts training: {}\n'.format(
                    step_id, time))

            self._update_class_weights()

            step_checkpoint_dir = os.path.join(
                self._checkpoint_dir,
                'step_id-{}'.format(step_id))
            step_summary_writer = tf.summary.FileWriter(
                step_checkpoint_dir, self._sess.graph)

            for iteration_id in tqdm(range(
                    self._num_iterations_per_step)):
                if ((iteration_id + 1) %
                        self._iteration_log_freq == 0):
                    with open(self._time_path, 'a') as f:
                        time = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S.%f')
                        f.write('step {} iteration {} '
                                'starts training: {}\n'.format(
                                    step_id, iteration_id, time))

                self._train_one_iteration(
                    summary_writer=step_summary_writer,
                    iteration_id=iteration_id)

                if ((iteration_id + 1) %
                        self._extra_iteration_checkpoint_freq == 0):
                    extra_checkpoint_dir = os.path.join(
                        step_checkpoint_dir,
                        'iteration_id-{}'.format(iteration_id))
                    self._save(
                        iteration_id=iteration_id,
                        step_id=step_id,
                        saver=tf.train.Saver(),
                        checkpoint_dir=extra_checkpoint_dir,
                        with_model=True,
                        with_states=True,
                        with_datasets=False)

                if ((iteration_id + 1) %
                        self._iteration_log_freq == 0):
                    with open(self._time_path, 'a') as f:
                        time = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S.%f')
                        f.write('step {} iteration {} '
                                'ends training: {}\n'.format(
                                    step_id, iteration_id, time))
            self._save(
                iteration_id=iteration_id,
                step_id=step_id,
                saver=tf.train.Saver(),
                checkpoint_dir=step_checkpoint_dir,
                with_model=True,
                with_states=True,
                with_datasets=True)
            self._save(
                iteration_id=iteration_id,
                step_id=step_id,
                saver=saver,
                with_model=True,
                with_states=True,
                with_datasets=True)

            with open(self._time_path, 'a') as f:
                time = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S.%f')
                f.write('step {} ends: {}\n'.format(step_id, time))

            if self._request_usage >= self._request_budget:
                break

            step_id += 1
