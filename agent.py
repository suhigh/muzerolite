import random
import time
import tensorflow as tf
import numpy as np
from math import isnan
import grpc

import network
from game import Game, GameHistory
from exceptions import MuZeroImplementationError
from utils import random_id, MinMaxStats

# For type annotations
from typing import Optional, Dict, List, Tuple, Union
from collections import Counter

from muzero_types import Player, Action, Policy, Value, Observation, ObservationBatch, ActionBatch
from environment import Environment
from network import MuProverNetwork, NetworkOutput, Network
from config import MuZeroConfig
from training import ActingStats
from reanalyse_buffer import ReanalyseBuffer
from shared_storage import SharedStorage
from protos import shared_storage_pb2_grpc, shared_storage_pb2


class Agent:
    """
    An agent playing a game in an Environment.
    """

    def __init__(self, config: MuZeroConfig, agent_id: Optional[str] = None) -> None:
        self.config: MuZeroConfig = config
        self.agent_id: str = agent_id if agent_id else 'agent_'+random_id()
        self.stats: ActingStats = ActingStats(config)

    """
    Default implementation just for training
    """
    def play_game(self, environment: Environment) -> GameHistory:
        start = time.time()

        history = GameHistory()
        if self.stats.should_reanalyse():
            sampled_history = self.get_analyse_game_history()
            history.reanalyse = True
            history.states = sampled_history.states
            history.rewards = sampled_history.rewards
            history.actions = sampled_history.actions

            for i, e in enumerate(history.actions):
                self.analyse_move(history, index=i)
                self.stats.state_added(history)

        else:
            game = Game(environment=environment)
            history = game.history
            while not game.terminal():
                # game.environment.env.render()
                action = self.make_move(game)
                game.apply(action)
                self.stats.state_added(history)

        self.stats.game_finished(history)
        end = time.time()

        history.metadata['agent_id'] = self.agent_id
        history.metadata['timing'] = end-start
        history.metadata.update(self.fill_metadata())
        return history

    def make_move(self, game: Game, training: bool = True) -> Action:
        """
        Choose a move to play in the Game.
        """
        raise MuZeroImplementationError('make_move', 'Agent')

    def fill_metadata(self) -> Dict[str, str]:
        """
        Sub-classed agents can use this callback to add further metadata to saved games.
        """
        return {}

    def get_analyse_game_history(self) -> Optional[GameHistory]:
        """
        Get a sample game from reanalyse buffer
        """
        raise MuZeroImplementationError('get_analyse_game_history', 'Agent')

    def analyse_move(self, history: GameHistory, index: int):
        """
        for each action, analyse the value and policy network
        """
        raise MuZeroImplementationError('analyse_move', 'Agent')


class RandomAgent(Agent):
    """
    Completely random agent, for testing purposes.
    """

    def make_move(self, game: Game, training: bool = True) -> Action:
        legal_actions = game.legal_actions()
        policy = np.zeros(game.environment.action_space_size)
        policy[legal_actions] = 1 / len(legal_actions)
        a, b = self.config.value_config.known_bounds.endpoints() if self.config.value_config.known_bounds else (0, 1)
        value = a + (a - b)*random.random()
        game.store_search_statistics(Value(value), Policy(tf.constant(policy)))
        return random.choice(legal_actions)

    def get_analyse_game_history(self) -> Optional[GameHistory]:
        return None


class NetworkAgent(Agent):
    """
    Agent choosing the next action according to a network's policy outputs.
    This is roughly like MCTSAgent with num_simulations = 0.
    """

    def __init__(self, config: MuZeroConfig, network: Network, reanalyse_buffer: Optional[ReanalyseBuffer] = None, temperature: float = 0.0, debug: bool = False) -> None:
        super().__init__(config=config)
        self.network: Network = network
        self.temperature: float = temperature
        self.debug: bool = debug
        self.reanalyse_buffer: ReanalyseBuffer = reanalyse_buffer

    def make_move(self, game: Game, training: bool = True) -> Action:
        observation_batch = ObservationBatch(tf.expand_dims(game.history.make_image(), axis=0))
        batch_network_output = self.network.initial_inference(observation_batch)
        network_output = batch_network_output.split_batch()[0]

        legal_actions = game.legal_actions()
        policy = network_output.masked_policy(legal_actions)
        game.store_search_statistics(network_output.value, Policy(tf.constant(policy)))

        if self.temperature == 0:
            _, action = max(zip(policy[legal_actions], legal_actions))
            return action
        else:
            weights = policy[legal_actions] ** (1 / self.temperature)
            return random.choices(legal_actions, weights=weights, k=1)[0]

    def get_analyse_game_history(self) -> Optional[GameHistory]:
        if self.reanalyse_buffer:
            return self.reanalyse_buffer.sample_game_history()
        else:
            return None


class Node:
    def __init__(self, parent: Optional['Node'] = None, prior: float = 1.0, to_play: Optional[Player] = 0) -> None:
        self.parent: Optional[Node] = parent
        self.prior: float = prior
        self.children: Dict[Action, Node] = {}
        self.hidden_state: Optional[Observation] = None
        self.reward: Optional[Value] = None
        self.to_play: Optional[Player] = to_play

        self.value_sum: Value = Value(0.0)
        self.visit_count: int = 0
        self.value: Value = Value(float('nan'))

    def expanded(self) -> bool:
        return len(self.children) > 0

    def update_value(self, value: Value) -> Value:
        self.value_sum += value
        self.visit_count += 1
        self.value = Value(self.value_sum / self.visit_count)
        return self.value

    def print(self, _prefix: str = '', name: str = 'Root', _last: bool = True) -> None:
        print(_prefix, '`- ' if _last else '|- ',
              f'{name}-{self.visit_count}: prior={self.prior:.2f}; value={self.value:.4f}', sep="")
        _prefix += '   ' if _last else '|  '
        child_count = len(self.children)
        for i, (action, child) in enumerate(self.children.items()):
            _last = i == (child_count - 1)
            child.print(_prefix, action, _last)


class MCTSAgent(Agent):
    """
    Use Monte-Carlo Tree-Search to select moves.
    """
    def __init__(self,
                 config: MuZeroConfig,
                 initial_network: MuProverNetwork,
                 reanalyse_buffer: Optional[ReanalyseBuffer] = None,
                 agent_id: Optional[str] = None,
                 debug: bool = False,
                 ) -> None:
        super().__init__(config=config, agent_id=agent_id)
        self.config: MuZeroConfig = config
        self.debug: bool = debug

        self.effective_discount: float = self.config.game_config.discount
        if config.game_config.num_players == 2:
            self.effective_discount *= -1
        self.reanalyse_buffer = reanalyse_buffer
        self.ref_network: Network = initial_network
        if reanalyse_buffer:
            channel = grpc.insecure_channel(config.training_config.client_storage_ip_port)
            self.storage_sub = shared_storage_pb2_grpc.ModelAgentStub(channel)
            mi: shared_storage_pb2.ModelInfo = self.storage_sub.latest_model(shared_storage_pb2.Empty())
            self.current_model_time = mi.time
            self.current_model_path = mi.path
            self.stats.start_training = mi.reanalyse_start

    @staticmethod
    def expand_node(node: Node, actions: List[Action], network_output: NetworkOutput, sampled: bool = False, sampling_k: int = 0, sampling_temperature: float = 1.0) -> None:
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward

        if sampled:
            policy = tf.pow(network_output.policy_logits, 1-1/sampling_temperature) / sampling_k
            network_output.policy_logits = policy

        policy = network_output.masked_policy(actions)
        for action, p in zip(actions, policy):
            node.children[action] = Node(prior=p, parent=node, to_play=Player(1-node.to_play))



    @staticmethod
    def softmax_sample(distribution: List[Tuple[int, Action]], temperature: float) -> Tuple[int, Action]:
        if temperature == 0.0:
            return max(distribution)
        else:
            # weights = [count ** (1 / temperature) for count, action in distribution]
            # return random.choices(distribution, weights=weights, k=1)[0]
            counts_exp = [np.exp(visit_counts) * (1 / temperature) for visit_counts, _ in distribution]
            probs = counts_exp / np.sum(counts_exp, axis=0)
            return random.choices(distribution, weights=probs, k=1)[0]

    def add_exploration_noise(self, node: Node) -> None:
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.mcts_config.root_dirichlet_alpha] * len(actions))
        frac = self.config.mcts_config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def select_leaf(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[Action, Node]:
        action = Action(-1)
        while node.expanded():
            action, node = self.select_child(node, min_max_stats)
        return action, node

    def select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[Action, Node]:
        _, action, child = max((self.ucb_score(child, min_max_stats), a, child) for a, child in node.children.items())
        return action, child

    def ucb_score(self, node: Node, min_max_stats: MinMaxStats) -> float:
        exploitation_score = self.config.mcts_config.default_value if isnan(node.value) else node.reward + self.effective_discount * node.value
        exploration_score = node.prior * self.config.exploration_function(node.parent.visit_count, node.visit_count)
        return min_max_stats.normalize(exploitation_score) + exploration_score

    def backpropagate(self, node: Node, value: Value, min_max_stats: MinMaxStats) -> None:
        # leaf_to_play = node.to_play
        while node is not None:
            min_max_stats.update(node.update_value(value))
            value = node.reward + self.effective_discount*value if node.reward is not None else Value(float('nan'))
            node = node.parent

    def add_sample_exploration_noise(self, policy: Policy) -> List:

        if self.config.mcts_config.sampling_noise != 0.0:
            noise = tf.constant(np.random.dirichlet([self.config.mcts_config.root_dirichlet_alpha] * len(self.config.action_space())), dtype=tf.float32)
            frac = self.config.mcts_config.root_exploration_fraction
            policy = (policy * (1 - frac) + noise * frac) / self.config.mcts_config.sampling_noise
        else:
            return tf.math.top_k(policy, self.config.mcts_config.sampling_k).indices.numpy()

        exp_policy = tf.math.exp(policy)
        policy_sum = tf.reduce_sum(exp_policy)
        return np.random.choice(self.config.action_space(), size=self.config.mcts_config.sampling_k, replace=False, p=(exp_policy / policy_sum).numpy())

    def run_mcts(self, root: Node, min_max_stats: MinMaxStats) -> None:

        for _ in range(self.config.mcts_config.num_simulations):
            action, leaf = self.select_leaf(root, min_max_stats)

            batch_hidden_state = ObservationBatch(tf.expand_dims(leaf.parent.hidden_state, axis=0))
            batch_action = ActionBatch(tf.constant([action]))
            batch_network_output = self.ref_network.recurrent_inference(batch_hidden_state, batch_action)
            network_output = batch_network_output.split_batch()[0]
            # sampled muzero implements
            if self.config.mcts_config.sampled:
                sampled_action = self.add_sample_exploration_noise(network_output.policy_logits)
                self.expand_node(node=leaf,
                                 actions=sampled_action,
                                 network_output=network_output,
                                 sampled=True,
                                 sampling_k=self.config.mcts_config.sampling_k)
            else:
                self.expand_node(node=leaf,
                                 actions=self.config.action_space(),
                                 network_output=network_output)
            self.backpropagate(leaf, network_output.value, min_max_stats)

    def select_action(self, node: Node, num_moves: int) -> Action:
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
        t = self.config.visit_softmax_temperature_fn(num_moves=num_moves, training_steps=self.ref_network.training_steps())
        _, action = self.softmax_sample(visit_counts, t)
        return action

    def make_move(self, game: Game, training: bool = True) -> Action:
        root = Node(to_play=Player(0))
        min_max_stats = MinMaxStats(known_bounds=self.config.value_config.known_bounds)
        observation = ObservationBatch(tf.expand_dims(game.history.make_image(), axis=0))
        self.expand_node(node=root,
                         actions=game.legal_actions(),
                         network_output=self.ref_network.initial_inference(observation).split_batch()[0])
        if training:
            self.add_exploration_noise(root)

        # with tf.device('/device:GPU:0'):
        #     self.run_mcts(root, min_max_stats)
        self.run_mcts(root, min_max_stats)
        action_space = self.config.action_space()
        policy = [root.children[a].visit_count / root.visit_count if a in root.children else 0 for a in action_space]
        if not training:
            print(policy)
        game.store_search_statistics(root.value, Policy(tf.constant(policy)))
        return self.select_action(root, len(game.history))

    def analyse_move(self, history: GameHistory, index: int):
        root = Node(to_play=Player(0))
        min_max_stats = MinMaxStats(known_bounds=self.config.value_config.known_bounds)
        observation = ObservationBatch(tf.expand_dims(history.make_image(index), axis=0))
        self.expand_node(node=root,
                         actions=history.states[index].legal_actions,
                         network_output=self.ref_network.initial_inference(observation).split_batch()[0])
        self.add_exploration_noise(root)
        # with tf.device('/device:GPU:0'):
        #     self.run_mcts(root, min_max_stats)
        self.run_mcts(root, min_max_stats)
        action_space = self.config.action_space()
        policy = [root.children[a].visit_count / root.visit_count if a in root.children else 0 for a in action_space]

        # update the values and policy
        history.policies.append(Policy(tf.constant(policy)))

        # the value has to be recalculated from the td manner, from the time line T'
        history.root_values.append(root.value)

    def fill_metadata(self) -> Dict[str, str]:
        return {'network_id': str(self.ref_network.training_steps())}

    def get_analyse_game_history(self) -> Optional[GameHistory]:
        if self.reanalyse_buffer:
            return self.reanalyse_buffer.sample_game_history()

    def update_network_to_current(self):
        mi: shared_storage_pb2.ModelInfo = self.storage_sub.latest_model(shared_storage_pb2.Empty())
        if mi.time == self.current_model_time:
            return

        self.current_model_path = mi.path
        self.current_model_time = mi.time
        self.stats.start_training = mi.reanalyse_start

        saved_network: MuProverNetwork = self.config.make_uniform_network()
        saved_network.checkpoint.restore(self.current_model_path)
        self.ref_network = saved_network


class PlayWithManMCTSAgent(MCTSAgent):

    def __init__(self, config: MuZeroConfig,
                 initial_network: Network,
                 agent_id: Optional[str] = None,
                 debug=False
                 ):
        config.mcts_config.temperature = 0.0
        super().__init__(config=config, initial_network=initial_network, agent_id=agent_id, debug=debug)

    def make_move(self, game: Game, training: bool = True) -> Action:
        game.debug = self.debug
        print(f'当前有效动作：{game.legal_actions()}')
        return super().make_move(game, training)

    def play_game(self, environment: Environment) -> GameHistory:
        # start = time.time()
        game = Game(environment=environment)
        while not game.terminal():
            # game.environment.env.render()

            # a = input('your choice:')
            # action = Action(int(a))
            # game.apply(action)
            action = self.make_move(game, False)
            print(f'Computer Performing {action}')
            game.apply(action)

            if not game.terminal():
                # action = self.make_move(game, False)
                # print(f'Computer Performing {action}')
                # game.apply(action)
                self.make_move(game, False)
                a = input('your choice:')
                action = Action(int(a))
                game.apply(action)

        # end = time.time()

        # game.history.metadata['agent_id'] = self.agent_id
        # game.history.metadata['timing'] = end-start
        # game.history.metadata.update(self.fill_metadata())
        return game.history


class MCTSAgentEvaluation(MCTSAgent):

    def __init__(self, config: MuZeroConfig,
                 shared_storage: SharedStorage,
                 agent_id: Optional[str] = None,
                 debug=False
                 ):
        super().__init__(config=config, shared_storage=shared_storage, agent_id=agent_id, debug=debug)

    def play_game(self, environment: Environment) -> GameHistory:
        start = time.time()
        game = Game(environment=environment)
        while not game.terminal():
            self.config.mcts_config.temperature = 0.0
            action = self.make_move(game, False)
            game.apply(action)

            if not game.terminal():
                self.config.mcts_config.temperature = 1.0
                action = self.make_move(game, True)
                game.apply(action)

        end = time.time()

        game.history.metadata['agent_id'] = self.agent_id
        game.history.metadata['timing'] = end-start
        game.history.metadata.update(self.fill_metadata())
        return game.history
