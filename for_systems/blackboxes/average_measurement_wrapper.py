import numpy as np
from tqdm import tqdm
import functools

from .blackbox import Blackbox

AVERAGING_FUNCTIONS = {
    'median': functools.partial(np.median, axis=0),
    'mean': functools.partial(np.mean, axis=0),
    'min': functools.partial(np.amin, axis=0)
}


class AverageMeasurement(Blackbox):
    def __init__(self, blackbox, num_measurements, averaging_method='median',
                 input_definition=None, num_warm_ups=0):
        self._blackbox = blackbox
        self._num_measurements = num_measurements
        self._averaging_method = averaging_method
        if not self._averaging_method in AVERAGING_FUNCTIONS:
            raise ValueError('Unkown averaging method {}'.format(
                self._averaging_method))
        self._input_definition = input_definition
        self._num_warm_ups = num_warm_ups
        if self._num_warm_ups > 0 and input_definition is None:
            raise ValueError('input_definition should be provided for enabling'
                             ' warm ups')

    def query(self, *args, **kwargs):
        if self._num_warm_ups > 0:
            warm_up_inputs = \
                self._input_definition.uniformly_sample_field_dict(
                    self._num_warm_ups,
                    progress_bar=False)
            self._blackbox.query(warm_up_inputs)

        results = []
        for i in tqdm(range(self._num_measurements)):
            result = self._blackbox.query(*args, **kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            results.append(result)
        results = list(zip(*results))
        for i in range(len(results)):
            results[i] = AVERAGING_FUNCTIONS[self._averaging_method](
                results[i])
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
