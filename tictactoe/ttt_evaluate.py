import tensorflow as tf
import os
from tictactoe import TicTacToeNetwork, TicTacToeEnvironment, make_config
from agent import MCTSAgentEvaluation


def run_eval(agent: MCTSAgentEvaluation, network: TicTacToeNetwork, env: TicTacToeEnvironment, eval_episodes: int):
    """Evaluate MuZero without noise added to the prior of the root and without softmax action selection"""
    returns = []
    for _ in range(eval_episodes):
        env.reset()
        history = agent.play_game(env)
        print(history)
        returns.append(sum(history.rewards))
    return sum(returns) / eval_episodes if eval_episodes else 0


def play():
    saved_models = 'F:/muzerottt3/models'
    config = make_config()
    config.mcts_config.temperature = 0.0
    env = TicTacToeEnvironment()
    td_model = '300000'

    model = tf.keras.models.load_model(os.path.join(saved_models, config.network_config.INITIAL_INFERENCE, td_model))

    network = TicTacToeNetwork(config, config.network_config.network_parameters.get('regularizer'),
                               config.network_config.network_parameters.get('hidden_state_size'),
                               config.network_config.network_parameters.get('hidden_size'))
    network.initial_inference_model = model

    model_recurrent = tf.keras.models.load_model(os.path.join(saved_models, config.network_config.RECURRENT_INFERENCE, td_model))
    network.recurrent_inference_model = model_recurrent

    agent = MCTSAgentEvaluation(config=config, network=network, debug=False)

    score = run_eval(agent, network, env, 100)
    print(f'评估得分：{score}')


if __name__ == '__main__':
    play()
