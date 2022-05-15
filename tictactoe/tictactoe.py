import numpy as np
import tensorflow as tf
import os
import logging

from config import MuZeroConfig, GameConfig, ReplayBufferConfig, ReanalyseBufferConfig, MCTSConfig, NetworkConfig, TrainingConfig, ScalarConfig
from utils import KnownBounds
from environment import Environment
from exceptions import MuZeroEnvironmentError
from network import MuProverNetwork, one_hot_tensor_encoder, dummy_model

# For type annotations
from typing import Optional, Tuple, List
from muzero_types import State, Value, Action, Observation, Player


def make_config() -> MuZeroConfig:
    game_config = GameConfig(name='TicTacToe',
                             environment_class=TicTacToeEnvironment,
                             environment_parameters={},
                             action_space_size=9,
                             num_players=2,
                             discount=1.0
                             )

    replay_buffer_config = ReplayBufferConfig(window_size=int(2e4),
                                              prefetch_buffer_size=10
                                              )

    reanalyse_buffer_config = ReanalyseBufferConfig(
        capacity=int(1e4),
        reanalyse_fraction=0.60
    )

    mcts_config = MCTSConfig(max_moves=9,
                             root_dirichlet_alpha=0.25,
                             root_exploration_fraction=0.25,
                             num_simulations=20,
                             temperature=0.8,
                             freezing_moves=9,
                             default_value=Value(0.0),
                             sampled=True,
                             sampling_temperature=8.0,
                             sampling_k=5,
                             sampling_noise=0.6
                             )

    network_config = NetworkConfig(network_class=TicTacToeNetwork,
                                   regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   hidden_state_size=128,
                                   hidden_size=128
                                   )

    training_config = TrainingConfig(optimizer=tf.keras.optimizers.Adam(),
                                     batch_size=128,
                                     training_steps=int(5e4),
                                     # training_steps=int(2e3),
                                     checkpoint_interval=int(5e2),
                                     replay_buffer_loginterval=50,
                                     num_unroll_steps=3,
                                     td_steps=6,
                                     steps_per_execution=1,
                                     lr_init=0.03,
                                     lr_decay_rate=0.9,
                                     lr_decay_steps=int(50000),
                                     momentum=0.9,
                                     # path='/home/wjb/ttttraining/',
                                     path='F:/muzerottt16/',
                                     start_epoch_cnt=1,
                                     server_storage_ip_port='[::]:50050',
                                     server_training_ip_port='[::]:50051',
                                     client_training_ip_port='127.0.0.1:50051',
                                     client_storage_ip_port='127.0.0.1:50050',
                                     interleave_range=2,
                                     waiting_replay_window=1500
                                     )
    # training_config.optimizer = tf.keras.optimizers.SGD(learning_rate=training_config.lr_init, momentum=training_config.momentum)
    training_config.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    reward_config = ScalarConfig(known_bounds=KnownBounds(minv=Value(0.0), maxv=Value(1.0)),
                                 support_size=None,
                                 loss_decay=0.0)

    value_config = ScalarConfig(known_bounds=KnownBounds(minv=None, maxv=Value(1.0)),
                                support_size=None,
                                loss_decay=4.0)

    return MuZeroConfig(game_config=game_config,
                        replay_buffer_config=replay_buffer_config,
                        reanalyse_buffer_config=reanalyse_buffer_config,
                        mcts_config=mcts_config,
                        training_config=training_config,
                        network_config=network_config,
                        value_config=value_config,
                        reward_config=reward_config)


class TicTacToeEnvironment(Environment):
    """
    The environment class of tic-tac-toe.
    """
    NO_REWARD: Value = Value(0.0)
    WINNING_MOVE: Value = Value(1.0)

    def __init__(self) -> None:
        super().__init__(action_space_size=9, num_players=2)

        # Game state
        self.board: Optional[np.ndarray] = None  # numpy array of shape (3,3), -1 if empty or player_id if filled
        self.to_play: Optional[int] = None       # player_id for the player who has to move next
        self.steps: Optional[int] = None         # number of steps taken in this episode
        self.ended: Optional[bool] = None        # flag to mark the episode has finished

    def __repr__(self) -> str:
        colors = {-1: '.', 0: 'X', 1: 'O'}
        result = ''
        if self.board is not None:
            for i in range(3):
                for j in range(3):
                    result += colors[self.board[i, j]]
                result += '\n'
        return result

    def _is_legal_action(self, action: Action) -> bool:
        if self.ended or action not in range(9):
            return False
        row = action // 3
        col = action % 3
        return self.board[row, col] == -1

    def _legal_actions(self) -> List[Action]:
        return [Action(i) for i in range(9) if self._is_legal_action(Action(i))]

    def _get_state(self) -> State:
        observation = tf.stack([tf.where(self.board == 0, 1., 0.),
                                tf.where(self.board == 1, 1., 0.),
                                self.to_play * tf.ones(shape=(3, 3))], axis=-1)
        return State(Observation(observation), Player(self.to_play), self._legal_actions())

    def _compute_reward(self) -> Value:
        for i in range(3):
            if np.all(self.board[i, :] == self.to_play) or np.all(self.board[:, i] == self.to_play):
                self.ended = True
        if np.all(self.board.diagonal() == self.to_play) or np.all(np.flip(self.board, axis=0).diagonal() == self.to_play):
            self.ended = True
        return self.WINNING_MOVE if self.ended else self.NO_REWARD

    def step(self, action: Action) -> Tuple[State, Value, bool, dict]:
        if not self._is_legal_action(action):
            raise MuZeroEnvironmentError(f'cannot perform action {action}')

        # Make the move
        row = action // 3
        col = action % 3
        self.board[row, col] = self.to_play

        # Compute the reward
        reward = self._compute_reward()

        # Switch players
        self.to_play = 1-self.to_play

        # Increase step counter to detect draws
        self.steps += 1
        if self.steps == 9:
            self.ended = True

        return self._get_state(), reward, self.ended, {}

    def reset(self) -> State:
        self.board = -np.ones(shape=(3, 3), dtype=np.int32)
        self.steps = 0
        self.ended = False
        self.to_play = 0
        return self._get_state()


class TicTacToeNetwork(MuProverNetwork):
    """
    Neural networks for tic-tac-toe game.
    """

    def __init__(self,
                 config: MuZeroConfig,
                 regularizer: tf.keras.regularizers.Regularizer,
                 hidden_state_size: int,
                 hidden_size: int
                 ) -> None:
        """
        Representation input (observation batch):       (batch_size, 3, 3, 3).
        Representation output (hidden state batch):     (batch_size, 3, 3, hidden_state_size)

        Encoded action batch:                           (batch_size, 3, 3, hidden_state_size+9)

        Dynamics input:                                 (batch_size, 3, 3, hidden_state_size+9)
        Dynamics outputs:
            - hidden_state:                             (batch_size, 3, 3, hidden_state_size)
            - reward:                                   (batch_size, )

        Prediction input:                               (batch_size, 3, 3, hidden_state_size)
        Prediction outputs:
            - policy_logits:                            (batch_size, action_space_size=9)
            - value:                                    (batch_size, )
        """

        random_tictactoe_state_preprocessing: tf.keras.Model = dummy_model(input_shape=(3, 3, 3))

        random_tictactoe_representation: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=3, padding='same', activation='relu',
                                   input_shape=(3, 3, 3), kernel_regularizer=regularizer, bias_regularizer=regularizer),
            tf.keras.layers.Dense(units=hidden_state_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
        ], name=config.network_config.REPRESENTATION)

        encoded_state_action = tf.keras.Input(shape=(3, 3, hidden_state_size + 9))
        x = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=3, padding='same', activation='relu',
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(encoded_state_action)
        hidden_state = tf.keras.layers.Dense(units=hidden_state_size, activation='relu',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        x = tf.keras.layers.Flatten()(hidden_state)
        reward_output = tf.keras.layers.Dense(units=1, activation='tanh', name='reward',
                                              kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        random_tictactoe_dynamics: tf.keras.Model = tf.keras.Model(inputs=encoded_state_action,
                                                                   outputs=[hidden_state, reward_output],
                                                                   name=config.network_config.DYNAMICS)

        hidden_state = tf.keras.Input(shape=(3, 3, hidden_state_size))
        x = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=3, activation='relu',
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(hidden_state)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=hidden_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        value_output = tf.keras.layers.Dense(units=1, activation='tanh', name='value',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        policy_logits_output = tf.keras.layers.Dense(units=9, name='policy_logits',
                                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

        random_tictactoe_prediction: tf.keras.Model = tf.keras.Model(inputs=hidden_state,
                                                                     outputs=[value_output, policy_logits_output],
                                                                     name=config.network_config.PREDICTION)

        super().__init__(config=config,
                         representation=random_tictactoe_representation,
                         dynamics=random_tictactoe_dynamics,
                         prediction=random_tictactoe_prediction,
                         state_action_encoder=one_hot_tensor_encoder(state_shape=(3, 3), action_space_size=9),
                         state_preprocessing=random_tictactoe_state_preprocessing)


def get_initial_network(config: MuZeroConfig) -> MuProverNetwork:
    start_ckpt = 'ckpt-' + str(config.training_config.start_epoch_cnt)

    network = TicTacToeNetwork(config, config.network_config.network_parameters.get('regularizer'),
                               config.network_config.network_parameters.get('hidden_state_size'),
                               config.network_config.network_parameters.get('hidden_size'))

    logging.info('representation')
    logging.info(network.representation.input_shape)
    logging.info(network.representation.output_shape)

    logging.info('dynamics')
    logging.info(network.dynamics.input_shape)
    logging.info(network.dynamics.output_shape)

    logging.info('prediction')
    logging.info(network.prediction.input_shape)
    logging.info(network.prediction.output_shape)

    logging.info('state_preprocessing')
    logging.info(network.state_preprocessing.input_shape)
    logging.info(network.state_preprocessing.output_shape)

    logging.info('recurrent_inference_model')
    logging.info(network.recurrent_inference_model.input_shape)
    logging.info(network.recurrent_inference_model.output_shape)

    logging.info('initial_inference_model')
    logging.info(network.initial_inference_model.input_shape)
    logging.info(network.initial_inference_model.output_shape)

    if os.path.exists(config.training_config.path + 'logs/ckpt/checkpoint'):
        network.checkpoint.restore(config.training_config.path + 'logs/ckpt/' + start_ckpt)
    return network
