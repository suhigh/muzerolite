from protos import shared_storage_pb2
from protos import shared_storage_pb2_grpc
import grpc
import logging
from concurrent.futures import ThreadPoolExecutor
from shared_storage import SharedStorage

from config import MuZeroConfig


class SharedStorageServer(shared_storage_pb2_grpc.ModelAgentServicer):
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.shared_storage = SharedStorage(config)

    def latest_model(self, request: shared_storage_pb2.Empty, context) -> shared_storage_pb2.ModelInfo:
        return self.shared_storage.latest_network()

    def update_current_generated_model(self, request: shared_storage_pb2.ModelInfo, context):
        logging.debug('updated the training network')
        self.shared_storage.save_network(request.path)
        return shared_storage_pb2.Empty()


def serve(config: MuZeroConfig) -> None:
    server = grpc.server(ThreadPoolExecutor())

    shared_storage_pb2_grpc.add_ModelAgentServicer_to_server(SharedStorageServer(config), server)
    server.add_insecure_port(config.training_config.server_storage_ip_port)
    server.start()
    logging.info("Shared Storage Server serving at %s", config.training_config.server_storage_ip_port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(None)
