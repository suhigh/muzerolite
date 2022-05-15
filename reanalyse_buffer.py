from typing import Callable, List, Dict, Tuple, Optional, Any
import random
from game import GameHistory
from config import MuZeroConfig
import pickle


class ReanalyseBuffer(object):

    def __init__(self):
        self.game_histories: List[GameHistory] = []

    def save_history(self, game_history: GameHistory):
        """ Saves a new game to the reanalyse buffer , to be reanalysed later ."""

    def sample_game_history(self) -> GameHistory:
        """ Samples a game that should be reanalysed ."""
        histories = random.choices(self.game_histories, weights=[len(h) for h in self.game_histories], k=1)
        return histories[0]

    def restore_buffer(self, file_path: str):
        """ restore the buffer from file"""

    def store_buffer(self, file_path: str):
        """ store the buffer to file"""


class DemonstrationBuffer(ReanalyseBuffer):
    """ A reanlayse buffer of a fixed set of demonstrations .
    Can be used to learn from existing policies , human demonstrations or for
    Offline RL.
    """
    def __init__(self, demonstrations: List[GameHistory]):
        super().__init__()
        self.game_histories.extend(demonstrations)


class MostRecentBuffer(ReanalyseBuffer):
    """ A reanalyse buffer that keeps the most recent games to reanalyse ."""
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self._capacity = config.reanalyse_buffer_config.capacity

    def save_history(self, game_history: GameHistory):
        self.game_histories.append(game_history)
        if len(self.game_histories) > self._capacity:
            self.game_histories.pop(0)


class HighestRewardBuffer(ReanalyseBuffer):
    """ A reanalyse buffer that keeps games with highest rewards to reanalyse ."""
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.capacity = config.reanalyse_buffer_config.capacity

    def save_history(self, game_history: GameHistory):
        self.game_histories.append(game_history)
        if len(self.game_histories) > self.capacity:
            self.game_histories.sort(key=lambda g: g.total_value, reverse=True)
            n = int(self.capacity/2)
            del self.game_histories[-n:]

