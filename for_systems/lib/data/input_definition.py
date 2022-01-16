import json
from tqdm import tqdm
import numpy as np

from .fields import FIELD_CLASS_MAP


class InputDefinition(object):
    def __init__(self, json_file_path=None):
        self._fields = []

        if json_file_path:
            self.load_from_json_file(json_file_path)

    @property
    def fields(self):
        return self._fields

    def __getitem__(self, key):
        return self._fields[key]

    def uniformly_sample_field_dict(self, num=None, progress_bar=True):
        if num is None:
            return_first = True
            num = 1
        else:
            return_first = False
        inputs = []
        iterator = range(num)
        if progress_bar:
            iterator = tqdm(iterator)
        for i in iterator:
            field_dict = {}
            for field in self._fields:
                field_dict[field.name] = field.uniformly_sample_field()
            inputs.append(field_dict)
        if return_first:
            return inputs[0]
        else:
            return inputs

    def uniformly_sample_numpy(self, num=None, progress_bar=True):
        if num is None:
            return_first = True
            num = 1
        else:
            return_first = False
        inputs = []
        iterator = range(num)
        if progress_bar:
            iterator = tqdm(iterator)
        for i in iterator:
            numpy = []
            for field in self._fields:
                numpy.append(field.uniformly_sample_numpy())
            numpy = np.concatenate(numpy, axis=0)
            inputs.append(numpy)
        inputs = np.stack(inputs, axis=0)
        if return_first:
            return inputs[0]
        else:
            return inputs

    def numpy_to_field_dict(self, numpy_inputs):
        num_inputs = numpy_inputs.shape[0]
        field_dict_inputs = []
        for i in range(num_inputs):
            field_dict = {}
            dim = 0
            for field in self._fields:
                field_dict[field.name] = field.numpy_to_field(
                    numpy_inputs[i, dim: dim + field.numpy_dim])
                dim += field.numpy_dim
            field_dict_inputs.append(field_dict)
        return field_dict_inputs

    def field_dict_to_numpy(self, field_dict_inputs):
        num_inputs = len(field_dict_inputs)
        total_dim = np.sum([field.numpy_dim for field in self._fields])
        numpy_inputs = np.zeros((num_inputs, total_dim), dtype=np.int32)
        for i in range(num_inputs):
            dim = 0
            for field in self._fields:
                numpy_inputs[i, dim: dim + field.numpy_dim] = \
                    field.field_to_numpy(field_dict_inputs[i][field.name])
                dim += field.numpy_dim
        return numpy_inputs

    def load_from_field_list(self, fields):
        self._fields = fields

    def dump_to_json_file(self, file_path):
        json_data = []
        for field in self._fields:
            json_data.append({
                'class': field.__class__.__name__,
                'args': field.serialize()})
        with open(file_path, 'w') as f:
            json.dump(json_data, f, sort_keys=True, indent=4)

    def load_from_json_file(self, file_path):
        self._fields = []
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            if not isinstance(json_data, list):
                raise ValueError('{} should contain a list'.format(file_path))
            for field in json_data:
                if not isinstance(field, dict):
                    raise ValueError('{} is not a dict'.format(str(field)))
                if not ('class' in field and 'args' in field):
                    raise ValueError(
                        '{} should contain "class" and "args"'.format(
                            str(field)))
                if field['class'] not in FIELD_CLASS_MAP:
                    raise ValueError(
                        'Unknown field class: {}'.format(field['class']))
                if not isinstance(field['args'], dict):
                    raise ValueError(
                        '{} is not a dict'.format(str(field['args'])))
                self._fields.append(
                    FIELD_CLASS_MAP[field['class']](**field['args']))
