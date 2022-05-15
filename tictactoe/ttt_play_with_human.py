import tensorflow as tf
import os
from tictactoe import TicTacToeNetwork, TicTacToeEnvironment, make_config
from agent import PlayWithManMCTSAgent


def play():
    # saved_models = 'F:/sht/ttttraining/logs/ckpt/ckpt-'
    saved_models = 'F:/muzerottt15/logs/ckpt/ckpt-'
    config = make_config()
    config.mcts_config.temperature = 0.0
    config.mcts_config.sampling_noise = 0.0
    env = TicTacToeEnvironment()
    td_model = '800'

    network = TicTacToeNetwork(config, config.network_config.network_parameters.get('regularizer'),
                               config.network_config.network_parameters.get('hidden_state_size'),
                               config.network_config.network_parameters.get('hidden_size'))

    network.checkpoint.restore(saved_models + td_model)
    print(f'test {network.steps.numpy()}')

    agent = PlayWithManMCTSAgent(config=config, initial_network=network, debug=True)

    history = agent.play_game(env)
    print(history)
    print([s.to_play for s in history.states])


def play_models():
    saved_models = 'F:/sht/ttttraining/models'
    config = make_config()
    config.mcts_config.temperature = 0.0
    config.mcts_config.sampling_noise = 0.0
    env = TicTacToeEnvironment()
    td_model = '100'

    model = tf.keras.models.load_model(os.path.join(saved_models, config.network_config.INITIAL_INFERENCE, td_model))

    network = TicTacToeNetwork(config, config.network_config.network_parameters.get('regularizer'),
                               config.network_config.network_parameters.get('hidden_state_size'),
                               config.network_config.network_parameters.get('hidden_size'))
    network.initial_inference_model = model

    model_recurrent = tf.keras.models.load_model(os.path.join(saved_models, config.network_config.RECURRENT_INFERENCE, td_model))
    network.recurrent_inference_model = model_recurrent

    agent = PlayWithManMCTSAgent(config=config, initial_network=network, debug=True)

    history = agent.play_game(env)
    print(history)
    print([s.to_play for s in history.states])


if __name__ == '__main__':
    play()
    # play_models()
