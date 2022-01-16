import os
import numpy as np
import tarfile
import _pickle as cPickle
from .dataset import Dataset


def _unpickle_cifar10(file):
    return cPickle.load(file, encoding='latin1')


def _construct_dataset(data_x, data_y, data_high_fraction):
    data_y[data_y >= 1] = 1
    data_y = 1 - data_y
    if data_high_fraction is not None:
        num_low = np.where(data_y == 0)[0].shape[0]
        num_high = int(
            num_low / (1. - data_high_fraction) * data_high_fraction)
        filter_ = data_y == 0
        ids = np.random.permutation(np.where(data_y == 1)[0])
        high_selected = ids[:num_high]
        filter_[high_selected] = 1
        data_x = data_x[filter_]
        data_y = data_y[filter_]
    print("num high={}, num low={}".format(
        np.where(data_y == 1)[0].shape[0],
        np.where(data_y == 0)[0].shape[0]))
    dataset = Dataset()
    dataset.load_from_data(data_x, data_y)
    return dataset


def load_data(dataset, data_high_fraction=None):
    if dataset == 'MNIST':
        f = open(os.path.join('data', 'MNIST', "train-images-idx3-ubyte"))
        loaded = np.fromfile(file=f, dtype=np.uint8)
        data_x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        data_x = data_x / 255.

        f = open(os.path.join('data', 'MNIST', 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=f, dtype=np.uint8)
        data_y = loaded[8:].reshape((60000)).astype(np.int32)

        return _construct_dataset(data_x, data_y, data_high_fraction)
    elif dataset == 'CIFAR10':
        tar = tarfile.open(os.path.join(
            'data', 'CIFAR10', "cifar-10-python.tar.gz"))
        data_x = []
        data_y = []
        for i in range(1, 6):
            file = tar.extractfile(
                os.path.join("cifar-10-batches-py", "data_batch_{}".format(i)))
            dict_ = _unpickle_cifar10(file)
            sub_data_x = dict_["data"]
            sub_data_y = np.asarray(dict_["labels"], dtype=np.int32)
            assert list(sub_data_x.shape) == [10000, 3072]
            assert sub_data_x.dtype == np.uint8
            assert list(sub_data_y.shape) == [10000]
            data_x.append(sub_data_x)
            data_y.append(sub_data_y)

        data_x = np.concatenate(data_x, axis=0)
        assert list(data_x.shape) == [50000, 3072]
        data_y = np.concatenate(data_y, axis=0)
        assert list(data_y.shape) == [50000]

        data_x = np.reshape(data_x, [50000, 3, 32, 32])
        data_x = np.transpose(data_x, [0, 2, 3, 1])
        data_x = data_x.astype(np.float64)
        data_x = data_x / 255. # -1~1
        return _construct_dataset(data_x, data_y, data_high_fraction)
