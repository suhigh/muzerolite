import tensorflow as tf
import os
from tictactoe import TicTacToeNetwork, TicTacToeEnvironment, make_config
from agent import PlayWithManMCTSAgent


def play():
    saved_models = 'F:/muzerottt4/models'
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

    agent = PlayWithManMCTSAgent(config=config, network=network, debug=True)

    history = agent.play_game(env)
    print(history)
    print(history.to_plays)


if __name__ == '__main__':
    play()
