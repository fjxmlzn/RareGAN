import tensorflow.compat.v1 as tf
import numpy as np
import os
import math
from tqdm import tqdm
import datetime
import sys
import imageio
import csv
import copy
import scipy.special


class RareGAN(object):
    def __init__(self, sess, dataset,
                 request_budget,
                 generator, discriminator, batch_size, z_dim,
                 balanced_disc_weights, class_loss_with_fake,
                 balanced_class_weights, high_fraction_multiple,
                 initial_random_budget, budget_per_step,
                 num_iterations_per_step,
                 checkpoint_dir, sample_dir, time_path,
                 extra_iteration_checkpoint_freq,
                 iteration_log_freq,
                 visualization_freq,
                 metric_callbacks=None, metric_freq=None, metric_path=None,
                 vis_num_h=10, vis_num_w=10,
                 gen_lr=0.0002, gen_beta1=0.5,
                 disc_lr=0.0002, disc_beta1=0.5,
                 disc_disc_coe=1.0, gen_disc_coe=1.0):
        self._sess = sess
        self._dataset = dataset
        self._request_budget = request_budget
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
        self._num_iterations_per_step = num_iterations_per_step
        self._checkpoint_dir = checkpoint_dir
        self._sample_dir = sample_dir
        self._time_path = time_path
        self._extra_iteration_checkpoint_freq = extra_iteration_checkpoint_freq
        self._iteration_log_freq = iteration_log_freq
        self._visualization_freq = visualization_freq
        self._metric_callbacks = metric_callbacks
        self._metric_freq = metric_freq
        self._metric_path = metric_path
        self._vis_num_h = vis_num_h
        self._vis_num_w = vis_num_w
        self._gen_lr = gen_lr
        self._gen_beta1 = gen_beta1
        self._disc_lr = disc_lr
        self._disc_beta1 = disc_beta1
        self._disc_disc_coe = disc_disc_coe
        self._gen_disc_coe = gen_disc_coe

        self._num_amplification_bins = 2

        self._image_dims = list(self._dataset.image_dims)
        print("image_dims: {}".format(self._image_dims))
        self._vis_random_latents = \
            self.gen_z(vis_num_h * vis_num_w)
        self._vis_random_labels = \
            np.mod(
                np.arange(vis_num_h * vis_num_w),
                self._num_amplification_bins)

        self._EPS = 1e-8
        self._MODEL_NAME = 'model'
        self._TRAINING_STATE_FILE_NAME = 'training_state.npz'

        if self._metric_callbacks is not None:
            for metric_callback in self._metric_callbacks:
                metric_callback.set_model(self)

    def build(self):
        self._build_connection()
        self._build_loss()
        self._build_summary()
        self._build_metric()
        self._saver = tf.train.Saver()

    def _build_metric(self):
        if self._metric_callbacks is not None:
            for metric_callback in self._metric_callbacks:
                metric_callback.build()

    def _build_connection(self):
        self._z_pl = tf.placeholder(
            tf.float32,
            [None, self._z_dim],
            name='z')
        self._z_for_class_pl = tf.placeholder(
            tf.float32,
            [None, self._z_dim],
            name='z_for_class')
        self._real_pl = tf.placeholder(
            tf.float32,
            [None] + self._image_dims,
            name='real')
        self._real_for_class_pl = tf.placeholder(
            tf.float32,
            [None] + self._image_dims,
            name='real_for_class')
        self._real_labels_for_class_pl = tf.placeholder(
            tf.int32,
            [None],
            name='real_labels_for_class')
        self._generated_labels_pl = tf.placeholder(
            tf.int32,
            [None],
            name='generated_labels')
        self._generated_labels_for_class_pl = tf.placeholder(
            tf.int32,
            [None],
            name='generated_labels_for_class')
        self._disc_class_weights_pl = tf.placeholder(
            tf.float32,
            [self._num_amplification_bins],
            name='disc_class_weights')
        self._class_class_weights_pl = tf.placeholder(
            tf.float32,
            [self._num_amplification_bins],
            name='class_class_weights')

        self._real_conditions_for_class = tf.one_hot(
            self._real_labels_for_class_pl,
            depth=self._num_amplification_bins)
        self._generated_conditions = tf.one_hot(
            self._generated_labels_pl,
            depth=self._num_amplification_bins)
        self._generated_conditions_for_class = tf.one_hot(
            self._generated_labels_for_class_pl,
            depth=self._num_amplification_bins)

        self._gen, _ = self._generator.build(
            self._z_pl, self._generated_conditions, train=True)

        self._gen_for_class, _ = self._generator.build(
            self._z_for_class_pl,
            self._generated_conditions_for_class,
            train=True)

        self._test_gen, _ = self._generator.build(
            self._z_pl, self._generated_conditions, train=False)

        self._disc_fake, self._disc_fake_class_logits, _ = \
            self._discriminator.build(self._gen, train=True)
        _, self._class_logits_fake, _ = self._discriminator.build(
            self._gen_for_class, train=True)

        self._disc_real, self._disc_real_class_logits, _ = \
            self._discriminator.build(self._real_pl, train=True)
        _, self._class_logits_real, _ = self._discriminator.build(
            self._real_for_class_pl, train=True)

        self._test_disc, self._test_class_logits, _ = \
            self._discriminator.build(self._real_pl, train=False)

        self._generator.print_layers()
        self._discriminator.print_layers()

    def _build_loss(self):
        batch_size = tf.shape(self._real_pl)[0]

        # Classification weights.
        if self._balanced_class_weights:
            class_id_fake = self._generated_labels_for_class_pl
            class_id_real = self._real_labels_for_class_pl
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
                labels=self._real_conditions_for_class) *
            self._class_real_weights)
        self._class_loss_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self._class_logits_fake,
                labels=self._generated_conditions_for_class) *
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
        ###
        self._disc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._disc_real, labels=tf.ones_like(self._disc_real)) *
            self._disc_real_weights)
        self._disc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._disc_fake,
                labels=tf.zeros_like(self._disc_fake))
            * self._disc_fake_weights)
        ###

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
                           self._disc_loss_class_real_normalized)

        # Generator losses.
        self._gen_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._disc_fake, labels=tf.ones_like(self._disc_fake)) *
            self._disc_fake_weights)
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
            'loss/disc', self._disc_loss))
        self._disc_summary.append(tf.summary.scalar(
            'disc/fake', tf.reduce_mean(tf.nn.sigmoid(self._disc_fake))))
        self._disc_summary.append(tf.summary.scalar(
            'disc/real', tf.reduce_mean(tf.nn.sigmoid(self._disc_real))))
        self._disc_summary.append(tf.summary.scalar(
            'disc/gap',
            tf.reduce_mean(
                tf.nn.sigmoid(self._disc_real) -
                tf.nn.sigmoid(self._disc_fake))))
        self._disc_summary.append(tf.summary.scalar(
            'disc/fake_weights', self._disc_fake_weights_mean))
        self._disc_summary.append(tf.summary.scalar(
            'disc/real_weights', self._disc_real_weights_mean))
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
                    self._real_pl:
                        numpy_inputs[i * self._batch_size:
                                     (i + 1) * self._batch_size]})
            disc.append(sub_disc)
            class_logits.append(sub_class_logits)
        return np.vstack(disc), np.vstack(class_logits)

    def sample_from(self, z, condition):
        outputs = []
        for i in range(int(math.ceil(float(z.shape[0]) / self._batch_size))):
            sub_outputs = self._sess.run(
                self._test_gen,
                feed_dict={
                    self._z_pl:
                        z[i * self._batch_size:
                          (i + 1) * self._batch_size],
                    self._generated_labels_pl:
                        condition[i * self._batch_size:
                                  (i + 1) * self._batch_size]})
            outputs.append(sub_outputs)
        return np.vstack(outputs)

    def sample(self, num_samples, condition=None):
        if condition is not None:
            if isinstance(condition, int):
                condition = np.asarray([condition] * num_samples)
            return self.sample_from(
                self.gen_z(num_samples),
                condition)
        else:
            return self.sample_from(
                self.gen_z(num_samples),
                self.gen_conditions(num_samples))

    def sample_high(self, num_samples):
        return self.sample(num_samples, condition=1)

    def gen_z(self, batch_size):
        return np.random.uniform(-1.0, 1.0, [batch_size, self._z_dim])

    def gen_labels(self, batch_size):
        _, labels = self._dataset.uniformly_sample_labeled(
            batch_size, condition_filter=-1)
        return labels

    def _add_random_data(self, num_samples, step):
        self._dataset.label_random_data(num_samples, step)
        self._request_usage += num_samples

    def _add_data(self, num_samples, step):
        numpy_inputs, numpy_inputs_id = self._dataset.get_unlabelled_data()
        _, class_logits = self.disc_from(numpy_inputs)
        softmax = scipy.special.softmax(class_logits, axis=1)
        max_ = np.max(softmax, axis=1)
        id_ = np.argsort(max_)
        print("Picking {} samples from {} samples".format(
            num_samples, numpy_inputs.shape[0]))
        print("The softmax of ranked {}th-sample is: {}".format(
            num_samples, softmax[id_[num_samples - 1]]))
        selected_numpy_inputs_id = numpy_inputs_id[id_[:num_samples]]
        self._dataset.label_data(selected_numpy_inputs_id, step, -2)
        self._request_usage += selected_numpy_inputs_id.shape[0]

        sys.stdout.flush()

    def _update_class_weights(self):
        high_fraction = self._dataset.get_class_fraction(1)

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

        batch_real_numpy_inputs, _ = \
            self._dataset.uniformly_sample(self._batch_size)

        batch_real_numpy_inputs_for_class, batch_real_labels_for_class = \
            self._dataset.uniformly_sample_labeled(self._batch_size)

        batch_generated_labels = self.gen_labels(
            self._batch_size)

        feed_dict = {
            self._z_pl: batch_z,
            self._z_for_class_pl: batch_z_for_class,
            self._real_pl: batch_real_numpy_inputs,
            self._real_for_class_pl:
                batch_real_numpy_inputs_for_class,
            self._real_labels_for_class_pl:
                batch_real_labels_for_class,
            self._generated_labels_pl:
                batch_generated_labels,
            self._generated_labels_for_class_pl:
                batch_real_labels_for_class,
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

    def _image_list_to_grid(self, image_list, num_row, num_col):
        assert num_row * num_col == image_list.shape[0]

        height, width, depth = self._image_dims
        image = np.zeros((num_row * height,
                          num_col * width,
                          depth))
        s_id = 0
        for row in range(num_row):
            for col in range(num_col):
                image[row * height: (row + 1) * height,
                      col * width: (col + 1) * width, :] = image_list[s_id]
                s_id += 1

        v_min = image.min() - self._EPS
        v_max = image.max() + self._EPS
        image = (image - v_min) / (v_max - v_min) * 255.0
        image = image.astype(np.uint8)

        print(v_min, v_max)

        return image

    def _visualize(self, step_id, iteration_id):
        samples = self.sample_from(
            self._vis_random_latents, self._vis_random_labels)
        image = self._image_list_to_grid(
            samples, self._vis_num_h, self._vis_num_w)
        file_path = os.path.join(
            self._sample_dir,
            "step_id-{},iteration_id-{}.png".format(
                step_id, iteration_id))
        imageio.imwrite(file_path, image)

    def _log_metric(self, step_id, iteration_id, summary_writer):
        if self._metric_callbacks is not None:
            metric = {}
            for metric_callback in self._metric_callbacks:
                metric.update(metric_callback.evaluate(step_id, iteration_id))
            if not os.path.isfile(self._metric_path):
                self._METRIC_FIELD_NAMES = ["step_id", "iteration_id"]
                for k in metric:
                    self._METRIC_FIELD_NAMES.append(k)
                with open(self._metric_path, "w") as csv_file:
                    writer = csv.DictWriter(
                        csv_file, fieldnames=self._METRIC_FIELD_NAMES)
                    writer.writeheader()
            elif not hasattr(self, "_METRIC_FIELD_NAMES"):
                with open(self._metric_path, "r") as csv_file:
                    reader = csv.DictReader(csv_file)
                    print("Load METRIC_FIELD_NAMES from the "
                          "existing metric file")
                    self._METRIC_FIELD_NAMES = reader.fieldnames

            with open(self._metric_path, "a") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=self._METRIC_FIELD_NAMES)
                data = {
                    "step_id": step_id,
                    "iteration_id": iteration_id}
                metric_string = copy.deepcopy(metric)
                for k in metric_string:
                    if isinstance(metric[k], (float, np.float32, np.float64)):
                        metric_string[k] = "{0:.12f}".format(metric_string[k])
                data.update(metric_string)
                writer.writerow(data)
            for k in metric:
                if isinstance(metric[k], (int, float, complex,
                                          np.float32, np.float64)):
                    summary = tf.Summary(
                        value=[tf.Summary.Value(
                            tag="metric/" + k, simple_value=metric[k])])
                    summary_writer.add_summary(summary, iteration_id)

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

        if self._metric_callbacks is not None:
            for metric_callback in self._metric_callbacks:
                metric_callback.load()

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
                    self._log_metric(
                        step_id, iteration_id, step_summary_writer)

                if (iteration_id + 1) % self._visualization_freq == 0:
                    self._visualize(step_id, iteration_id)

                if (((iteration_id + 1) % self._metric_freq == 0) or
                        (iteration_id == self._num_iterations_per_step - 1)):
                    self._log_metric(
                        step_id, iteration_id, step_summary_writer)

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
