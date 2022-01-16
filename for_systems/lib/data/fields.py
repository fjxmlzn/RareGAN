import numpy as np


class Field(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def numpy_dim(self):
        return self._numpy_dim

    def uniformly_sample_numpy(self):
        raise NotImplementedError

    def uniformly_sample_field(self):
        raise NotImplementedError

    def field_to_numpy(self, value):
        raise NotImplementedError

    def numpy_to_field(self, value):
        raise NotImplementedError

    def serialize(self):
        return {'name': self._name}


class BitField(Field):
    def __init__(self, num_bits, decimal, *args, **kwargs):
        super(BitField, self).__init__(*args, **kwargs)

        self._num_bits = num_bits
        self._decimal = decimal
        self._numpy_dim = self._num_bits

    @property
    def num_bits(self):
        return self._num_bits

    def _uniformly_sample_binary(self):
        return np.random.randint(low=0, high=2, size=self._num_bits)

    def _uniformly_sample_decimal(self):
        return int(np.random.randint(low=0, high=2**self._num_bits, size=1))

    def uniformly_sample_numpy(self):
        return self._uniformly_sample_binary()

    def uniformly_sample_field(self):
        if self._decimal:
            return self._uniformly_sample_decimal()
        else:
            return self._uniformly_sample_binary()

    def _binary_to_decimal(self, value):
        decimal = 0
        for i in range(len(value)):
            decimal = decimal * 2 + value[i]
        return decimal

    def _decimal_to_binary(self, value):
        binary = np.binary_repr(value)
        binary = list(map(int, binary))
        binary = [0] * (self._num_bits - len(binary)) + binary
        binary = np.asarray(binary)
        return binary

    def field_to_numpy(self, value):
        if self._decimal:
            return self._decimal_to_binary(value)
        else:
            return value

    def numpy_to_field(self, value):
        if self._decimal:
            return self._binary_to_decimal(value)
        else:
            return value

    def decimal_to_field(self, value):
        if self._decimal:
            return value
        else:
            return self._decimal_to_binary(value)

    def serialize(self):
        data = super(BitField, self).serialize()
        data.update({
            'num_bits': self._num_bits,
            'decimal': self._decimal})
        return data


class ChoiceField(Field):
    def __init__(self, choices, length, *args, **kwargs):
        super(ChoiceField, self).__init__(*args, **kwargs)

        if not isinstance(choices, list):
            raise ValueError('choices should be a list')
        self._choices = choices
        self._length = length
        self._numpy_dim = 1 if self._length is None else self._length

        self._choice_inverse_map = {
            choice: i for i, choice in enumerate(self._choices)}

    @property
    def choices(self):
        return self._choices

    def _uniformly_sample_id(self):
        return np.random.randint(
            low=0,
            high=len(self._choices),
            size=self._numpy_dim)

    def uniformly_sample_numpy(self):
        return self._uniformly_sample_id()

    def uniformly_sample_field(self):
        field = [self._choices[i] for i in list(self._uniformly_sample_id())]
        if self._length is None:
            return field[0]
        else:
            return field

    def field_to_numpy(self, value):
        if self._length is None:
            value = [value]
        return np.asarray([self._choice_inverse_map[i] for i in value])

    def numpy_to_field(self, value):
        field = [self._choices[i] for i in list(value)]
        if self._length is None:
            return field[0]
        else:
            return field

    def serialize(self):
        data = super(ChoiceField, self).serialize()
        data.update({
            'choices': self._choices,
            'length': self._length})
        return data


class FixedField(Field):
    def __init__(self, value, *args, **kwargs):
        super(FixedField, self).__init__(*args, **kwargs)
        self._value = value
        self._numpy_dim = 0

    @property
    def value(self):
        return self._value

    def uniformly_sample_numpy(self):
        return np.asarray([])

    def uniformly_sample_field(self):
        return self._value

    def field_to_numpy(self, value):
        return np.asarray([])

    def numpy_to_field(self, value):
        return self._value

    def serialize(self):
        data = super(FixedField, self).serialize()
        data.update({'value': self._value})
        return data


_FIELD_CLASSES = [BitField, ChoiceField, FixedField]

FIELD_CLASS_MAP = {
    field_class.__name__: field_class for field_class in _FIELD_CLASSES
}
