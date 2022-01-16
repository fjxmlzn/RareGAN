import numpy as np
import time

from .blackbox import Blackbox

class PacketClassificationBlackbox(Blackbox):
    def __init__(self, decision_tree):
        self._decision_tree = decision_tree

    def query(self, field_dict_inputs):
        packets = self._binary_to_int(field_dict_inputs)
        times = self._get_classification_time(packets, self._decision_tree)
        return times

    def _get_classification_time(self, packets, tree):

        times = []
        cnt = 0
        
        for packet in packets:
            tic = time.clock()
            actual_match = tree.match(packet)
            toc = time.clock()
            
            if actual_match is None:
                cnt += 1
                
            time_elapsed = round((toc - tic), 8) * 1000
            times.append(time_elapsed)

        return np.asarray(times)

    def _binary_to_int(self, inputs):
        packets = []
        for p in inputs:
            packets.append((self._b2i(p['src_ip']), self._b2i(p['dst_ip']), 
                self._b2i(p['src_port']), self._b2i(p['dst_port']),
                self._b2i(p['proto'])))
        return packets

    def _b2i(self, b_list):
        return int("".join(str(x) for x in b_list), 2) 
