import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight as sklearn_compute_class_weight
from collections import Counter


class Dataset(object):
    def __init__(self, folder=None):
        self._numpy_inputs = None
        self._amplifications = None
        self._steps = None
        self._conditions = None

        self._FILE_NAME = "data.npz"

        if folder:
            self.load_from_folder(folder)

        self._thresholds = None
        self._class_balanced_sampling_weights = None

    @property
    def num_bins(self):
        return self._num_bins

    @property
    def numpy_inputs(self):
        return self._numpy_inputs

    @property
    def amplifications(self):
        return self._amplifications

    @property
    def conditions(self):
        return self._conditions

    @property
    def steps(self):
        return self._steps

    @property
    def num_samples(self):
        if self._numpy_inputs is None:
            return 0
        else:
            return self._numpy_inputs.shape[0]

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds):
        self._thresholds = thresholds

    def _compute_class_balanced_sampling_weights(self):
        if not self._thresholds:
            raise ValueError('Please set thresholds first')
        class_ = np.digitize(self._amplifications, self._thresholds)
        class_weight = sklearn_compute_class_weight(
            'balanced', range(len(self._thresholds) + 1), class_)
        class_weight = class_weight.astype(np.float64)
        self._class_balanced_sampling_weights = class_weight[class_]
        self._class_balanced_sampling_weights = \
            (self._class_balanced_sampling_weights /
             np.sum(self._class_balanced_sampling_weights))

    def add_data(self, numpy_inputs, amplifications, steps, conditions):
        num_samples = [numpy_inputs.shape[0],
                       amplifications.shape[0],
                       steps.shape[0],
                       conditions.shape[0]]
        if not (len(set(num_samples)) == 1):
            raise ValueError('The number of samples are not matched: '
                             '{}'.format(num_samples))
        if self._numpy_inputs is None:
            self._numpy_inputs = numpy_inputs
            self._amplifications = amplifications
            self._steps = steps
            self._conditions = conditions
        else:
            if not (numpy_inputs.shape[1:] == self._numpy_inputs.shape[1:]):
                raise ValueError('"numpy_inputs" has wrong dimensions')
            self._numpy_inputs = np.vstack([self._numpy_inputs, numpy_inputs])
            self._amplifications = np.concatenate(
                [self._amplifications, amplifications])
            self._steps = np.concatenate([self._steps, steps])
            self._conditions = np.concatenate([self._conditions, conditions])

        # Reset balanced sampling weights.
        self._class_balanced_sampling_weights = None

    def merge(self, dataset):
        self.add_data(
            numpy_inputs=dataset.numpy_inputs,
            amplifications=dataset.amplifications,
            steps=dataset.steps,
            conditions=dataset.conditions)

        # Reset balanced sampling weights.
        self._class_balanced_sampling_weights = None

    def shrink(self, num_samples):
        self._numpy_inputs = self._numpy_inputs[:num_samples]
        self._amplifications = self._amplifications[:num_samples]
        self._steps = self._steps[:num_samples]
        self._conditions = self._conditions[:num_samples]

        # Reset balanced sampling weights.
        self._class_balanced_sampling_weights = None

    def _amplification_to_class_labels(self, amplifications):
        bins = np.digitize(amplifications, self._thresholds)
        return np.eye(self._num_bins, dtype=np.float32)[bins]

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
        if not self._thresholds:
            raise ValueError('Please set thresholds first')
        if self._conditions is None:
            # No data.
            return np.zeros(shape=len(self._thresholds) + 1)
        if condition_filter:
            ids = np.where(self._conditions == condition_filter)[0]
        else:
            ids = np.arange(self._conditions.shape[0])
        class_ = np.digitize(self._amplifications[ids], self._thresholds)
        """
        # Add one sample to each class
        class_ = np.concatenate(
            [class_, np.arange(len(self._thresholds) + 1)])
        """

        class_weight = self._compute_class_weights(
            classes=range(len(self._thresholds) + 1),
            y=class_,
            last_fraction=last_fraction)
        return class_weight

    def uniformly_sample(self, batch_size, condition_filter=None):
        if condition_filter:
            candidates = np.where(self._conditions == condition_filter)[0]
        else:
            candidates = self._numpy_inputs.shape[0]
        batch_ids = np.random.choice(candidates, size=batch_size)
        numpy_inputs = self._numpy_inputs[batch_ids]
        amplifications = self._amplifications[batch_ids]
        return numpy_inputs, amplifications

    def class_balanced_sample(self, batch_size):
        if self._class_balanced_sampling_weights is None:
            # Lazy weight computation.
            self._compute_class_balanced_sampling_weights()
        batch_ids = np.random.choice(
            self._numpy_inputs.shape[0],
            size=batch_size,
            p=self._class_balanced_sampling_weights)
        numpy_inputs = self._numpy_inputs[batch_ids]
        amplifications = self._amplifications[batch_ids]
        return numpy_inputs, amplifications

    def get_amplification_at_fraction(
            self, fraction, step_filter=None, condition_filter=None):
        filter_ = np.ones(self._amplifications.shape[0], dtype=bool)
        if step_filter is not None:
            filter_ = filter_ * (self._steps == step_filter)
        if condition_filter is not None:
            filter_ = filter_ * (self._conditions == condition_filter)
        amplifications = self._amplifications[filter_]
        sorted_amplifications = np.sort(amplifications)[::-1]
        id_ = int(fraction * (sorted_amplifications.shape[0] - 1))
        result = sorted_amplifications[id_]
        print('Getting amplification at fraction {} based on {} samples: '
              '{:.6f}'.format(
                  fraction, amplifications.shape[0], result))
        return result

    def get_amplification_fraction(
            self, amplification, step_filter=None, condition_filter=None):
        filter_ = np.ones(self._amplifications.shape[0], dtype=bool)
        if step_filter is not None:
            filter_ = filter_ * (self._steps == step_filter)
        if condition_filter is not None:
            filter_ = filter_ * (self._conditions == condition_filter)
        amplifications = self._amplifications[filter_]
        result = (
            float(np.where(amplifications >= amplification)[0].shape[0]) /
            amplifications.shape[0])
        print('Getting fraction of amplification {} based on {} samples'
              ': {:.6f}'.format(
                  amplification, amplifications.shape[0], result))
        return result

    def get_amplification_min_rank(self, rank, step_filter=None,
                                   condition_filter=None):
        filter_ = np.ones(self._amplifications.shape[0], dtype=bool)
        if step_filter is not None:
            filter_ = filter_ * (self._steps == step_filter)
        if condition_filter is not None:
            filter_ = filter_ * (self._conditions == condition_filter)
        amplifications = self._amplifications[filter_]
        num_samples = amplifications.shape[0]
        sorted_amplifications = sorted(list(set(list(amplifications))))
        num_amplifications = len(sorted_amplifications)
        result = sorted_amplifications[rank]
        print('Getting amplification at min rank {} based on {} samples and {}'
              ' amplification values: {:.6f}'.format(
                  rank, num_samples, num_amplifications, result))
        return result

    def dump_to_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez(
            os.path.join(folder, self._FILE_NAME),
            numpy_inputs=self._numpy_inputs,
            amplifications=self._amplifications,
            steps=self._steps,
            conditions=self._conditions)

    def load_from_folder(self, folder):
        data = np.load(os.path.join(folder, self._FILE_NAME),allow_pickle=True)
        self._numpy_inputs = data["numpy_inputs"]
        self._amplifications = data["amplifications"]
        self._steps = data["steps"]
        self._conditions = data["conditions"]
