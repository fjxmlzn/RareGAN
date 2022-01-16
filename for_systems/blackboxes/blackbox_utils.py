import time
import sys
import pickle

from .dns_blackbox import DNSBlackbox
from .rpc_wrapper import RPCClient
from .packet_classification_blackbox import PacketClassificationBlackbox
from .average_measurement_wrapper import AverageMeasurement
from .warm_up_measurement_wrapper import WarmUpMeasurement
from .random_order_measurement_wrapper import RandomOrderMeasurement
from .rpc_runner import RPCRunnerClient

import blackboxes.decision_tree.tree as tree
sys.modules['tree'] = tree


def create_blackbox_from_config(config, input_definition):
    if config['blackbox'] == 'dns':
        blackbox = DNSBlackbox(server_ip=config['dns_server_ip'])
        auxiliary = None
    elif config['blackbox'] == 'packet_classification':
        with open(config['tree_file'], 'rb') as f:
            tree_ = pickle.load(f)
        blackbox = PacketClassificationBlackbox(decision_tree=tree_)

        if 'num_warm_ups' in config:
            blackbox = WarmUpMeasurement(
                blackbox,
                input_definition=input_definition,
                num_warm_ups=config['num_warm_ups'])
        if 'random_order' in config and config['random_order']:
            blackbox = RandomOrderMeasurement(blackbox)
        rpc_runner = RPCRunnerClient(
            server_ip=config['rpc_server_ip'],
            server_port=config['rpc_server_port'])
        rpc_port = rpc_runner.start(blackbox)
        time.sleep(5)
        blackbox = RPCClient(
            server_ip=config['rpc_server_ip'],
            server_port=rpc_port)
        if 'num_measurements' in config:
            blackbox = AverageMeasurement(
                blackbox,
                num_measurements=config['num_measurements'],
                averaging_method=config['averaging_method'])
        auxiliary = (rpc_runner, rpc_port)
        # Warm up
        if ('num_pre_warm_ups' in config and config['num_pre_warm_ups'] > 0):
            field_dict_inputs = input_definition.uniformly_sample_field_dict(
                config['num_pre_warm_ups'])
            blackbox.query(field_dict_inputs)
    else:
        raise ValueError('Unknow blackbox: {}'.format(config['blackbox']))

    return blackbox, auxiliary


def close_blackbox_from_config(config, blackbox, auxiliary):
    if config['blackbox'] == 'packet_classification':
        rpc_runner, rpc_port = auxiliary
        rpc_runner.stop(rpc_port)
