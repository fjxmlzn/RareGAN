import numpy as np

from lib.data.dataset import Dataset


def amplification_to_class_labels(amplifications, thresholds):
    thresholds = sorted(thresholds)
    num_bins = len(thresholds) + 1
    bins = np.digitize(amplifications, thresholds)
    return np.eye(num_bins, dtype=np.float32)[bins]


def get_target_threshold_from_config(config, blackbox=None,
                                     input_definition=None, reevaluation=True):
    if 'tgt_thld' in config:
        return config['tgt_thld']
    else:
        dataset = Dataset(config['tgt_thld_ref_dataset'])
        if reevaluation:
            if blackbox is None or input_definition is None:
                raise ValueError('blackbox and input_definition should be '
                                 'provided')
            field_dict_inputs = input_definition.numpy_to_field_dict(
                dataset.numpy_inputs)
            amplifications = blackbox.query(field_dict_inputs)
            new_dataset = Dataset()
            new_dataset.add_data(
                numpy_inputs=dataset.numpy_inputs,
                amplifications=amplifications,
                steps=np.zeros(amplifications.shape[0], dtype=np.int32),
                conditions=np.zeros(amplifications.shape[0], dtype=np.int32))
        else:
            new_dataset = dataset
        return new_dataset.get_amplification_at_fraction(
            fraction=config['tgt_thld_fr'])
