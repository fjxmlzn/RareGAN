import numpy as np
import os
from collections import Counter


class Dataset(object):
    def __init__(self, folder=None):
        self._numpy_inputs = None
        self._labels = None
        self._steps = None
        self._conditions = None
        self._is_labeled = None
        self._image_dims = None

        self._FILE_NAME = "data.npz"
        self._num_classes = 2

        if folder:
            self.load_from_folder(folder)

    @property
    def image_dims(self):
        return self._image_dims

    @property
    def numpy_inputs(self):
        return self._numpy_inputs

    @property
    def labels(self):
        return self._labels

    @property
    def conditions(self):
        return self._conditions

    @property
    def steps(self):
        return self._steps

    @property
    def is_labeled(self):
        return self._is_labeled

    @property
    def num_samples(self):
        if self._numpy_inputs is None:
            return 0
        else:
            return self._numpy_inputs.shape[0]

    def label_data(self, data_ids, step, condition):
        self._is_labeled[data_ids] = 1
        self._steps[data_ids] = step
        self._conditions[data_ids] = condition

    def label_random_data(self, num_samples, step):
        data_ids = np.random.choice(
            self._numpy_inputs.shape[0],
            size=num_samples,
            replace=False)
        self.label_data(data_ids, step, -1)

    def _compute_class_weights(self, classes, y, last_fraction):
        counter = Counter(list(y))
        class_weight = []
        assert len(set(classes)) == len(classes)
        for i in classes:
            if i == classes[-1]:
                fraction = last_fraction
            else:
                fraction = (1. - last_fraction) / (len(classes) - 1)
            class_weight.append(len(y) * fraction / counter[i])
        return class_weight

    def compute_class_weights(self, last_fraction=0.5, condition_filter=None):
        if self._labels is None:
            # No data.
            return np.zeros(shape=self._num_classes)
        if condition_filter:
            filter_ = self._conditions == condition_filter
        else:
            filter_ = np.asarray([True] * self._labels.shape[0])
        filter_ = filter_ * (self._is_labeled == 1)
        class_ = self._labels[filter_]
        print('Getting class weights based on {} high samples and {} low '
              'samples'.format(
                  np.where(class_ == 1)[0].shape[0],
                  np.where(class_ == 0)[0].shape[0]))
        # Add one sample to each class
        #class_ = np.concatenate(
        #    [class_, np.arange(self._num_classes)])

        class_weight = self._compute_class_weights(
            classes=range(self._num_classes),
            y=class_,
            last_fraction=last_fraction)
        return class_weight

    def get_class_fraction(self, label):
        all_count = np.where(self._is_labeled == 1)[0].shape[0]
        filtered_count = np.where(
            (self._labels == label) * (self._is_labeled == 1))[0].shape[0]
        result = float(filtered_count) / all_count
        print('Getting fraction of class {}: {}/{}={}'.format(
            label, filtered_count, all_count, result))
        return result

    def uniformly_sample(self, batch_size):
        batch_ids = np.random.choice(
            self._numpy_inputs.shape[0],
            size=batch_size)
        numpy_inputs = self._numpy_inputs[batch_ids]
        labels = self._labels[batch_ids]
        return numpy_inputs, labels

    def uniformly_sample_labeled(self, batch_size, label_filter=None,
                                 condition_filter=None):
        filter_ = self._is_labeled == 1
        if label_filter is not None:
            filter_ = filter_ * (self._labels == label_filter)
        if condition_filter:
            filter_ = filter_ * (self._conditions == condition_filter)
        candidates = np.where(filter_)[0]
        print("Sampling from {} candidates".format(candidates.shape[0]))
        batch_ids = np.random.choice(candidates, size=batch_size)
        numpy_inputs = self._numpy_inputs[batch_ids]
        labels = self._labels[batch_ids]
        return numpy_inputs, labels

    def get_unlabelled_data(self):
        candidates = np.where(self._is_labeled == 0)[0]
        return self._numpy_inputs[candidates], candidates

    def get_data(self, label):
        return self._numpy_inputs[self._labels == label]

    def dump_to_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez(
            os.path.join(folder, self._FILE_NAME),
            numpy_inputs=self._numpy_inputs,
            labels=self._labels,
            steps=self._steps,
            conditions=self._conditions,
            is_labeled=self._is_labeled)

    def load_from_folder(self, folder):
        data = np.load(os.path.join(folder, self._FILE_NAME))
        self._numpy_inputs = data["numpy_inputs"]
        self._labels = data["labels"]
        self._steps = data["steps"]
        self._conditions = data["conditions"]
        self._is_labeled = data["is_labeled"]
        self._image_dims = self._numpy_inputs.shape[1:]

    def load_from_data(self, numpy_inputs, labels):
        self._numpy_inputs = numpy_inputs
        self._labels = labels
        self._steps = np.ones(numpy_inputs.shape[0]) * (-1)
        self._conditions = np.ones(numpy_inputs.shape[0]) * (-3)
        self._is_labeled = np.zeros(numpy_inputs.shape[0], dtype=np.int32)
        self._image_dims = self._numpy_inputs.shape[1:]
