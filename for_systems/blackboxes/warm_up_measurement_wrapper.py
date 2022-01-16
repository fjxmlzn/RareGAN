from .blackbox import Blackbox


class WarmUpMeasurement(Blackbox):
    def __init__(self, blackbox, input_definition, num_warm_ups):
        self._blackbox = blackbox
        self._input_definition = input_definition
        self._num_warm_ups = num_warm_ups

    def query(self, *args, **kwargs):
        warm_up_inputs = self._input_definition.uniformly_sample_field_dict(
            self._num_warm_ups,
            progress_bar=False)
        if len(args) > 0:
            args = list(args)
            args[0] = warm_up_inputs + args[0]
            args = tuple(args)
        elif 'field_dict_inputs' in kwargs:
            kwargs['field_dict_inputs'] = (
                warm_up_inputs + kwargs['field_dict_inputs'])
        else:
            raise ValueError('field_dict_inputs is not found in the input '
                             'parameters')
        results = self._blackbox.query(*args, **kwargs)
        if not isinstance(results, tuple):
            results = [results]
        else:
            results = list(results)
        for i in range(len(results)):
            results[i] = results[i][self._num_warm_ups:]
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
