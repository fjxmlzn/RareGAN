import six

from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class Blackbox():

    @abstractmethod
    def query(self, field_dict_inputs):
        pass
