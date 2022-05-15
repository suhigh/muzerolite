from protos import training_runtime_pb2_grpc, training_runtime_pb2
from typing import Iterable
import grpc
import logging
from concurrent.futures import ThreadPoolExecutor
from replay_buffer import ReplayBuffer
from game_services import history_from_protobuf


class TrainingRuntimeServicer(training_runtime_pb2_grpc.TrainingRuntimeServicer):
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    def SaveHistory(self, request: training_runtime_pb2.GameHistory, context: grpc.ServicerContext) -> training_runtime_pb2.NumGamesResponse:
        history = history_from_protobuf(request)

        self.replay_buffer.save_history(history)
        return training_runtime_pb2.NumGamesResponse(num_games=1)

    def SaveMultipleHistory(self, request_iterator: Iterable[training_runtime_pb2.GameHistory], context) -> training_runtime_pb2.NumGamesResponse:
        pass


def serve(address: str, training_callback, config) -> None:
    replay_buffer = ReplayBuffer(config)
    server = grpc.server(ThreadPoolExecutor())
    training_runtime_pb2_grpc.add_TrainingRuntimeServicer_to_server(TrainingRuntimeServicer(replay_buffer), server)
    server.add_insecure_port(address)
    server.start()
    logging.info("Training Runtime Server serving at %s", address)
    # server.wait_for_termination()
    training_callback(*(config, replay_buffer))
    server.stop(None)
