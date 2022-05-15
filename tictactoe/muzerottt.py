import os
import time

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tictactoe import TicTacToeNetwork, TicTacToeEnvironment, make_config
from agent import MCTSAgent
from replay_buffer import ReplayBuffer
from reanalyse_buffer import MostRecentBuffer
from ttt_training import ttt_model, ttt_train_network, TTTRecentBuffer
from shared_storage import SharedStorage
import threading


class Playing(threading.Thread):
    def __init__(self, agent: MCTSAgent, replay_buffer: ReplayBuffer, env: TicTacToeEnvironment, thread_id: int):
        threading.Thread.__init__(self)
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.train_finished = False
        self.env = env
        self.count = 0
        self.reanalyse_count = 0
        self.wait_to_play = True
        self.thread_id = thread_id

    def start_training(self):
        self.wait_to_play = False
        self.agent.stats.start_training = True

    def run(self) -> None:
        while not self.train_finished:
            self.count += 1
            self.agent.update_network_to_current()
            start = time.time()
            game_history = self.agent.play_game(self.env)
            print(f'{self.thread_id}: {time.time()-start}')

            if self.count % 300 == 0:
                timing = game_history.metadata['timing']
                num_moves = len(game_history)
                print(f'cnt:{self.count}:Game played in {timing:.2f}s, {num_moves} moves ==> {timing / num_moves:.2f}s per move, including reanalyse history cnt:{self.reanalyse_count}!')
                print(game_history)
            self.replay_buffer.save_history(game_history)
            if not game_history.is_reanalyse():
                self.agent.reanalyse_buffer.save_history(game_history)
            else:
                self.reanalyse_count += 1




