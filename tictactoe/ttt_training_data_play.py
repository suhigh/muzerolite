import os
import time
import grpc

from tictactoe import TicTacToeEnvironment, make_config, get_initial_network
from agent import MCTSAgent
from game_services import history_to_protobuf
from ttt_training import TTTRecentBuffer
import threading
from protos import training_runtime_pb2_grpc

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Playing(threading.Thread):
    def __init__(self, agent: MCTSAgent, env: TicTacToeEnvironment):
        threading.Thread.__init__(self)
        self.agent = agent
        self.train_finished = False
        self.env = env
        self.count = 0
        self.reanalyse_count = 0

        channel = grpc.insecure_channel(agent.config.training_config.client_training_ip_port)
        self.training_sub = training_runtime_pb2_grpc.TrainingRuntimeStub(channel)

    def run(self) -> None:
        while not self.train_finished:
            self.count += 1
            self.agent.update_network_to_current()
            # start = time.time()
            game_history = self.agent.play_game(self.env)
            # print(f'{time.time()-start}, reanalyse: {self.agent.stats.start_training}')

            if self.count % 300 == 0:
                timing = game_history.metadata['timing']
                num_moves = len(game_history)
                print(f'cnt:{self.count}:Game played in {timing:.2f}s, {num_moves} moves ==> {timing / num_moves:.2f}s per move, including reanalyse history cnt:{self.reanalyse_count}!')
                print(game_history)

            self.training_sub.SaveHistory(history_to_protobuf(game_history))
            if not game_history.is_reanalyse():
                self.agent.reanalyse_buffer.save_history(game_history)
            else:
                self.reanalyse_count += 1


def to_play():

    config = make_config()
    network = get_initial_network(config)

    reanalyse_buffer = TTTRecentBuffer(config)

    agent_x = MCTSAgent(config=config, initial_network=network, reanalyse_buffer=reanalyse_buffer)
    env_x = TicTacToeEnvironment()
    env_x.reset()
    player_x = Playing(agent=agent_x, env=env_x)
    player_x.start()


if __name__ == '__main__':
    to_play()
