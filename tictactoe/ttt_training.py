import pickle
from training import *
from network import MuProverNetwork
from tensorflow.keras import Model
from reanalyse_buffer import HighestRewardBuffer, MostRecentBuffer
from tensorboard.plugins.hparams import api as hp
from tictactoe import make_config, get_initial_network
import logging
import time
import os
from training_runtime_service import serve


def ttt_model(
        config: MuZeroConfig,
        network: MuProverNetwork,
        optimizer: tf.keras.optimizers.Optimizer) -> tf.keras.Model:
    unrolled_model = build_unrolled_model(config, network)
    unrolled_model.compile(
        loss={
            config.network_config.UNROLLED_REWARDS: config.reward_config.loss,
            config.network_config.UNROLLED_VALUES: config.value_config.loss,
            config.network_config.UNROLLED_POLICY_LOGITS: tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            config.network_config.UNROLLED_REWARDS: config.reward_config.loss_decay,
            config.network_config.UNROLLED_VALUES: config.value_config.loss_decay,
            config.network_config.UNROLLED_POLICY_LOGITS: 1.0
        },
        optimizer=optimizer,
        steps_per_execution=config.training_config.steps_per_execution,
        # metrics={config.network_config.UNROLLED_REWARDS: 'mse', config.network_config.UNROLLED_VALUES: 'mse', config.network_config.UNROLLED_POLICY_LOGITS:'categorical_accuracy'}
    )
    return unrolled_model


def ttt_train_network(config: MuZeroConfig,
                      network: MuProverNetwork,
                      replay_buffer: ReplayBuffer,
                      unrolled_model: Model,
                      saved_models_path: str,
                      writer: Optional[tf.summary.SummaryWriter] = None,
                      checkpoint_manager: Optional[tf.train.CheckpointManager] = None) -> Dict[str, List[float]]:

    replay_buffer_loginterval = config.training_config.replay_buffer_loginterval
    dataset = replay_buffer.as_dataset(batch_size=config.training_config.batch_size)

    muzero_callback = MuZeroCallback(training_network=network,
                                     saved_models_path=saved_models_path,
                                     checkpoint_manager=checkpoint_manager)
    callbacks = [muzero_callback]
    if writer:
        loss_logger = LossLoggerCallback(config=config, network=network, writer=writer)
        callbacks.append(loss_logger)
        if replay_buffer_loginterval is not None:
            replay_buffer_callback = ReplayBufferLoggerCallback(
                network=network,
                replay_buffer=replay_buffer,
                replay_buffer_loginterval=replay_buffer_loginterval,
                writer=writer)
            callbacks.append(replay_buffer_callback)

    num_epochs = config.training_config.training_steps // config.training_config.checkpoint_interval
    with tf.device('/device:GPU:0'):
        history = unrolled_model.fit(dataset,
                                     epochs=num_epochs,
                                     steps_per_epoch=config.training_config.checkpoint_interval,
                                     callbacks=callbacks)
        return history.history


class TTTHighestRewardBuffer(HighestRewardBuffer):
    """ A reanalyse buffer that keeps games with highest rewards to reanalyse ."""
    def __init__(self, config: MuZeroConfig):
        super().__init__(config)
        self.zero_index: int = 0
        self.pos1_index: int = 0
        self.keys: set = set()

    def save_history(self, game_history: GameHistory):
        key = int(''.join(map(str, game_history.actions)))
        if key in self.keys:
            return
        else:
            self.keys.add(key)

        t = game_history.total_value
        if t == -1:
            self.zero_index += 1
            self.pos1_index += 1
            self.game_histories.insert(self.zero_index-1, game_history)
        elif t == 0:
            self.pos1_index += 1
            self.game_histories.insert(self.pos1_index-1, game_history)
        else:
            self.game_histories.append(game_history)

        if len(self.game_histories) > self.capacity:
            del self.game_histories[0]
            if self.zero_index > 0:
                self.zero_index -= 1
            if self.pos1_index > 0:
                self.pos1_index -= 1

    def sample_game_history(self) -> GameHistory:
        """ Samples a game that should be reanalysed ."""
        histories = random.choices(self.game_histories, k=1)
        his_cp = GameHistory()
        his_cp.actions = histories[0].actions
        his_cp.states = histories[0].states
        his_cp.rewards = histories[0].rewards
        return his_cp

    def restore_buffer(self, file_path: str):
        with open(file_path, 'rb') as f:
            loaded_history = pickle.load(f)
            self.game_histories = loaded_history
            self.pos1_index = 0
            self.zero_index = 0
            for h in loaded_history:
                key = int(''.join(map(str, h.actions)))
                self.keys.add(key)
                if h.total_value == 0:
                    self.pos1_index += 1
                elif h.total_value == -1:
                    self.pos1_index += 1
                    self.zero_index += 1

    def store_buffer(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self.game_histories, f)


class TTTRecentBuffer(MostRecentBuffer):
    """ A reanalyse buffer that keeps games with highest rewards to reanalyse ."""
    def __init__(self, config: MuZeroConfig):
        super().__init__(config)
        self.keys: set = set()

    def save_history(self, game_history: GameHistory):
        key = int(''.join(map(str, game_history.actions)))
        if key in self.keys:
            return
        else:
            self.keys.add(key)

        t = game_history.total_value
        self.game_histories.append(game_history)

        if len(self.game_histories) > self._capacity:
            del self.game_histories[0]

    def sample_game_history(self) -> GameHistory:
        """ Samples a game that should be reanalysed ."""
        histories = random.choices(self.game_histories, k=1)
        his_cp = GameHistory()
        his_cp.actions = histories[0].actions
        his_cp.states = histories[0].states
        his_cp.rewards = histories[0].rewards
        return his_cp

    def restore_buffer(self, file_path: str):
        with open(file_path, 'rb') as f:
            loaded_history = pickle.load(f)
            self.game_histories = loaded_history
            for h in loaded_history:
                key = int(''.join(map(str, h.actions)))
                self.keys.add(key)

    def store_buffer(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self.game_histories, f)


def tensorboard_model_summary(model: tf.keras.Model, line_length: int = 100) -> str:
    lines = []
    model.summary(print_fn=lambda line: lines.append(line), line_length=line_length)
    lines.insert(3, '-'*line_length)
    positions = [lines[2].find(col) for col in ['Layer', 'Output', 'Param', 'Connected']]
    positions.append(line_length)
    table = ['|'+'|'.join([line[positions[i]:positions[i+1]] for i in range(len(positions)-1)])+'|' for line in lines[2:-4] if line[0] not in ['=', '_']]
    result = '# Model summary\n' + '\n'.join(table) + '\n\n# Parameter summary\n' + '\n\n'.join(lines[-4:-1])
    return result


def start_train(config: MuZeroConfig, replay_buffer: ReplayBuffer):

    network = get_initial_network(config)

    saved_models = config.training_config.path+'models'
    logdir = config.training_config.path+'logs'

    checkpoint_path = os.path.join(logdir, 'ckpt') if logdir else None
    checkpoint_manager = tf.train.CheckpointManager(network.checkpoint, checkpoint_path, max_to_keep=5) if logdir else None

    if not os.path.exists(config.training_config.path+'logs/ckpt/checkpoint'):
        network.save_tfx_models(saved_models)

    writer = tf.summary.create_file_writer(logdir) if logdir else None
    if writer:
        hyperparameters = config.hyperparameters()
        with writer.as_default():
            hp.hparams(hyperparameters)
            tf.summary.text(name='Networks/Representation',
                            data=tensorboard_model_summary(network.representation),
                            step=0)
            tf.summary.text(name='Networks/Dynamics',
                            data=tensorboard_model_summary(network.dynamics),
                            step=0)
            tf.summary.text(name='Networks/Prediction',
                            data=tensorboard_model_summary(network.prediction),
                            step=0)
            tf.summary.text(name='Networks/Initial inference',
                            data=tensorboard_model_summary(network.initial_inference_model),
                            step=0)
            tf.summary.text(name='Networks/Recurrent inference',
                            data=tensorboard_model_summary(network.recurrent_inference_model),
                            step=0)

    model = ttt_model(config=config, network=network, optimizer=config.training_config.optimizer)

    if os.path.exists(config.training_config.path+'games.data'):
        replay_buffer.load_games(config.training_config.path+'games.data')

    while replay_buffer.total_games < config.training_config.waiting_replay_window:
        logging.info(f'{time.strftime("%d %H:%M:%S")}, games:{replay_buffer.total_games}')
        time.sleep(30)

    logging.info('start train')
    ttt_train_network(config=config, network=network, replay_buffer=replay_buffer, unrolled_model=model,
                      saved_models_path=saved_models, writer=writer, checkpoint_manager=checkpoint_manager)

    replay_buffer.save_games(config.training_config.path+'games.data')
    logging.info('train finished')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = make_config()
    serve(config.training_config.server_training_ip_port, training_callback=start_train, config=config)
