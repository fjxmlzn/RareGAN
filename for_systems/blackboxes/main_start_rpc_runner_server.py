import argparse
from .rpc_runner import RPCRunnerServer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start RPC runner server')
    parser.add_argument(
        '--rpc_port',
        type=int,
        default=6001)

    args = parser.parse_args()

    RPCRunnerServer.start(port=args.rpc_port)
