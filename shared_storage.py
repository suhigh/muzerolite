import time
import os

from threading import RLock
from config import MuZeroConfig
from protos import shared_storage_pb2


class SharedStorage(object):
    """
    provide the latest versions of the network checkpoint path.
    """

    def __init__(self, config: MuZeroConfig):
        self._network_lock: RLock = RLock()
        self._time_stamp: float = time.time()
        self.config: MuZeroConfig = config
        self._checkpoint_path: str = ''
        self._reanalyse_start: bool = False

        start_ckpt = 'ckpt-' + str(config.training_config.start_epoch_cnt)
        if os.path.exists(config.training_config.path + 'logs/ckpt/checkpoint'):
            if os.path.exists(config.training_config.path + 'logs/ckpt/' + start_ckpt):
                self._checkpoint_path = config.training_config.path + 'logs/ckpt/' + start_ckpt
                self._reanalyse_start = True

    def save_network(self, checkpoint_path: str):
        with self._network_lock:
            self._time_stamp = time.time()
            self._checkpoint_path = checkpoint_path
            self._reanalyse_start = True
        #     self._networks.clear()
        #     for i in range(self.player_qty):
        #         saved_network = self.__training_network.config.make_uniform_network()
        #         checkpoint = tf.train.Checkpoint(network=saved_network.checkpoint,
        #                                          optimizer=self.__training_network.config.training_config.optimizer)
        #         checkpoint.read(checkpoint_path)
        #         self._networks.append(saved_network)

    def latest_network(self) -> shared_storage_pb2.ModelInfo:
        mi: shared_storage_pb2.ModelInfo = shared_storage_pb2.ModelInfo()
        mi.path = self._checkpoint_path
        mi.time = self._time_stamp
        mi.reanalyse_start = self._reanalyse_start

        return mi
