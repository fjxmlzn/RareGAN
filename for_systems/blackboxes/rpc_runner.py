import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np
import multiprocessing
import threading
from .rpc_wrapper import RPCServer

PORT_POOL = list(range(49152, 65536))

_rpc_processes = {}
_lock = threading.Lock()


class RPCRunnerServer(rpyc.Service):
    def __init__(self):
        pass

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_start(self, blackbox):
        blackbox = rpyc.classic.obtain(blackbox)

        rpc_port = int(np.random.choice(PORT_POOL))

        print('Receiving start request at RPC port {}'.format(rpc_port))

        rpc_process = multiprocessing.Process(
            target=RPCServer.start,
            args=(blackbox, rpc_port))
        rpc_process.start()

        with _lock:
            _rpc_processes[rpc_port] = rpc_process

        return rpc_port

    def exposed_stop(self, rpc_port):
        print('Receiving stop request at RPC port {}'.format(rpc_port))
        with _lock:
            if rpc_port not in _rpc_processes:
                return False
            _rpc_processes[rpc_port].terminate()
            _rpc_processes[rpc_port].join()

            del _rpc_processes[rpc_port]
        return True

    @staticmethod
    def start(port):
        server = ThreadedServer(
            RPCRunnerServer,
            port=port,
            protocol_config={'sync_request_timeout': None,
                             'allow_pickle': True})
        server.start()


class RPCRunnerClient():
    def __init__(self, server_ip, server_port):
        self._server_ip = server_ip
        self._server_port = server_port

    def start(self, blackbox):
        rpc_server = rpyc.connect(
            self._server_ip,
            self._server_port,
            config={'sync_request_timeout': None,
                    'allow_pickle': True})
        port = rpyc.classic.obtain(rpc_server.root.start(
            blackbox=blackbox))
        rpc_server.close()
        return port

    def stop(self, port):
        rpc_server = rpyc.connect(
            self._server_ip,
            self._server_port,
            config={'sync_request_timeout': None,
                    'allow_pickle': True})
        flag = rpyc.classic.obtain(rpc_server.root.stop(port))
        rpc_server.close()
        return flag
