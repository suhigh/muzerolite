import tensorflow as tf
import os
from tictactoe import TicTacToeNetwork, TicTacToeEnvironment, make_config
from agent import MCTSAgentEvaluation
from muzero_types import Observation, ActionBatch, ValueBatch, PolicyBatch, Value, Action
from config import MuZeroConfig
from game import GameHistory
import random
from shared_storage import SharedStorage

td_steps = 5
effective_discount = -1
num_unroll_steps = 4
config = make_config()
action_space_size = 9

def compute_target_value(history: GameHistory, index: int) -> Value:
    bootstrap_index = index + td_steps
    if bootstrap_index < len(history):
        value = history.root_values[bootstrap_index] * effective_discount ** td_steps
    else:
        value = 0
    value += sum(
        reward * effective_discount ** i for i, reward in enumerate(history.rewards[index:bootstrap_index]))
    return Value(value)


def preprocess_history(history: GameHistory) -> None:
    # Extend actions past the terminal state using random actions
    extended_actions = history.actions.copy()
    extended_actions.extend([Action(random.randrange(action_space_size)) for _ in range(num_unroll_steps)])
    history.extended_actions = tf.constant(extended_actions, dtype=tf.int32)

    # Extend target rewards past the terminal state using null rewards
    target_rewards = history.rewards.copy()
    target_rewards.extend([0 for _ in range(num_unroll_steps)])
    history.target_rewards = config.reward_config.inv_to_scalar(tf.constant(target_rewards, dtype=tf.float32))

    # Extend target values past the terminal state using the last value
    target_values = [compute_target_value(history, index) for index in range(len(history))]
    target_values.extend([0 for _ in range(num_unroll_steps + 1)])
    history.target_values = config.value_config.inv_to_scalar(tf.constant(target_values, dtype=tf.float32))

    # Extend target policies past the terminal state using uniform policies
    history.target_policies = tf.concat(
        [
            tf.stack(history.policies),
            tf.ones(shape=(num_unroll_steps + 1, action_space_size)) / action_space_size
        ], axis=0)

    history.total_value = sum(reward * effective_discount ** i for i, reward in enumerate(history.rewards))
    history.metadata['num_batches'] = 0


def run_eval(agent: MCTSAgentEvaluation, network: TicTacToeNetwork, env: TicTacToeEnvironment, eval_episodes: int):
    """Evaluate MuZero without noise added to the prior of the root and without softmax action selection"""
    returns = []
    for _ in range(eval_episodes):
        env.reset()
        history = agent.play_game(env)
        # rs = sum(history.rewards)
        # print(history.rewards)
        # print(history.states)
        rs = sum([r if s.to_play == 0 else -r for r, s in zip(history.rewards, history.states)])
        returns.append(rs)
        preprocess_history(history)

        print(f'{history} : {rs} : {history.total_value}')
        print(f'values: {history.root_values}')
        print(f'reward {history.rewards}')
        print(f'policy {history.policies}')
        print(f'target value: {history.target_values}')
        print(f'target reward {history.target_rewards}')
        print(f'target policy {history.target_policies}')

    return sum(returns) / eval_episodes if eval_episodes else 0


def play():
    saved_models = 'F:/muzerottt12/models'
    config = make_config()
    config.mcts_config.temperature = 0.0
    config.mcts_config.sampling_temperature = 0.0
    env = TicTacToeEnvironment()
    td_model = '250'

    model = tf.keras.models.load_model(os.path.join(saved_models, config.network_config.INITIAL_INFERENCE, td_model))

    network = TicTacToeNetwork(config, config.network_config.network_parameters.get('regularizer'),
                               config.network_config.network_parameters.get('hidden_state_size'),
                               config.network_config.network_parameters.get('hidden_size'))
    network.initial_inference_model = model

    model_recurrent = tf.keras.models.load_model(os.path.join(saved_models, config.network_config.RECURRENT_INFERENCE, td_model))
    network.recurrent_inference_model = model_recurrent
    shared_storage = SharedStorage(network)
    agent = MCTSAgentEvaluation(config=config, shared_storage=shared_storage, debug=False)

    score = run_eval(agent, network, env, 100)
    print(f'评估得分：{score}')


if __name__ == '__main__':
    play()
