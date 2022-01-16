import rpyc
from rpyc.utils.server import ThreadedServer

from .blackbox import Blackbox


class RPCServer(rpyc.Service):
    def __init__(self, blackbox):
        if not isinstance(blackbox, Blackbox):
            raise ValueError('blackbox should be an instance of Blackbox')
        self._blackbox = blackbox

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_query(self, *args, **kwargs):
        args = rpyc.classic.obtain(args)
        kwargs = rpyc.classic.obtain(kwargs)
        return self._blackbox.query(*args, **kwargs)

    @staticmethod
    def start(blackbox, port):
        server = ThreadedServer(
            RPCServer(blackbox),
            port=port,
            protocol_config={'sync_request_timeout': None,
                             'allow_pickle': True})
        server.start()


class RPCClient(Blackbox):
    def __init__(self, server_ip, server_port):
        self._server_ip = server_ip
        self._server_port = server_port

    def query(self, *args, **kwargs):
        rpc_server = rpyc.connect(
            self._server_ip,
            self._server_port,
            config={'sync_request_timeout': None,
                    'allow_pickle': True})
        result = rpyc.classic.obtain(rpc_server.root.query(*args, **kwargs))
        rpc_server.close()
        return result
