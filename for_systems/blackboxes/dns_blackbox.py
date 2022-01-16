import signal
import dns.message
import dns.rdataclass
import dns.rdatatype
import dns.query
import dns.flags
import time
import multiprocessing
from tqdm import tqdm
from functools import partial
import numpy as np
import sys

from .blackbox import Blackbox


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _query_worker(field_dict, timeouts, server_ip, delay_between):
    """Modified from:
    https://github.com/soojinm/ampmap/blob/master/local_test/DNS_local/libs/blackbox.py
    """
    m = dns.message.Message()
    id_ = field_dict["id"]
    qr = field_dict["qr"]
    aa = field_dict["aa"]
    tc = field_dict["tc"]
    rd = field_dict["rd"]
    ra = field_dict["ra"]
    cd = field_dict["cd"]
    ad = field_dict["ad"]
    opcode = field_dict["opcode"]
    rcode = field_dict["rcode"]
    edns = field_dict["edns"]
    payload = field_dict["payload"]
    url = field_dict["url"]
    rdataclass = field_dict["rdataclass"]
    rdatatype = field_dict["rdatatype"]
    dnssec = field_dict["dnssec"]

    m.id = id_
    if qr:
        m.flags |= int(dns.flags.QR)
    if aa:
        m.flags |= int(dns.flags.AA)
    if tc:
        m.flags |= int(dns.flags.TC)
    if rd:
        m.flags |= int(dns.flags.RD)
    if ra:
        m.flags |= int(dns.flags.RA)
    if ad:
        m.flags |= int(dns.flags.AD)
    if cd:
        m.flags |= int(dns.flags.CD)
    m.set_opcode(int(opcode))
    m.set_rcode(int(rcode))
    m.edns = int(edns)
    m.payload = int(payload)
    if dnssec:
        m.ednsflags |= int(dns.flags.DO)
    qname = dns.name.from_text(url)
    m.find_rrset(m.question, qname, rdataclass, rdatatype, create=True,
                 force_unique=True)
    request_len = len(m.to_wire())

    for timeout in timeouts:
        try:
            response = dns.query.udp(m, server_ip, timeout=timeout)
            response_len = len(response.to_wire())
        except dns.exception.DNSException as ex:
            print("Exception {} for {} seconds: {}".format(
                type(ex).__name__, timeout, field_dict))
            sys.stdout.flush()
            response_len = 0

        if response_len > 0:
            time.sleep(delay_between)
            return float(response_len) / float(request_len)

        print("No response for {} seconds: {}".format(timeout, field_dict))
        sys.stdout.flush()
        time.sleep(timeout)

    print("No response: {}".format(field_dict))
    sys.stdout.flush()
    return 0


class DNSBlackbox(Blackbox):
    def __init__(self, server_ip,
                 timeouts=[0.01, 0.05, 1.0, 2.0, 4.0, 8.0],
                 num_process=1,
                 delay_between=0.02):
        self._server_ip = server_ip
        self._timeouts = timeouts
        self._num_process = num_process
        self._delay_between = delay_between

        self._pool = multiprocessing.Pool(
            processes=self._num_process,
            initializer=_ignore_sigint)

    def query(self, field_dict_inputs):
        amplifications = []
        for amplification in tqdm(self._pool.imap(
                partial(_query_worker,
                        timeouts=self._timeouts,
                        server_ip=self._server_ip,
                        delay_between=self._delay_between),
                field_dict_inputs),
                total=len(field_dict_inputs)):
            amplifications.append(amplification)
        return np.asarray(amplifications)
