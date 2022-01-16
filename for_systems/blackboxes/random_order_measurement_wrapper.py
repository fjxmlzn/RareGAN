import numpy as np

from .blackbox import Blackbox


class RandomOrderMeasurement(Blackbox):
    def __init__(self, blackbox):
        self._blackbox = blackbox

    def _random_permutation(self, field_dict_inputs):
        new_field_dict_inputs = [None] * len(field_dict_inputs)
        id_ = np.random.permutation(len(field_dict_inputs))
        for i in range(len(field_dict_inputs)):
            new_field_dict_inputs[id_[i]] = field_dict_inputs[i]
        return new_field_dict_inputs, id_

    def _revert_permutation(self, result, permutation):
        return result[permutation]

    def query(self, *args, **kwargs):
        if len(args) > 0:
            field_dict_inputs, permutation = self._random_permutation(args[0])
            args = list(args)
            args[0] = field_dict_inputs
            args = tuple(args)
        elif 'field_dict_inputs' in kwargs:
            kwargs['field_dict_inputs'], permutation = \
                self._random_permutation(kwargs['field_dict_inputs'])
        else:
            raise ValueError('field_dict_inputs is not found in the input '
                             'parameters')
        results = self._blackbox.query(*args, **kwargs)
        if not isinstance(results, tuple):
            results = [results]
        else:
            results = list(results)
        for i in range(len(results)):
            results[i] = self._revert_permutation(results[i], permutation)
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
