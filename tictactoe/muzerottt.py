import os
import time

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tictactoe import TicTacToeNetwork, TicTacToeEnvironment, make_config
from agent import MCTSAgent
from replay_buffer import ReplayBuffer
from ttt_training import ttt_model, ttt_train_network
import threading


class Playing(threading.Thread):
    def __init__(self, agent: MCTSAgent, replay_buffer: ReplayBuffer, env: TicTacToeEnvironment):
        threading.Thread.__init__(self)
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.train_finished = False
        self.env = env
        self.count = 0

    def run(self) -> None:
        while not self.train_finished:
            self.count += 1
            game_history = self.agent.play_game(self.env)

            if self.count % 300 == 0:
                timing = game_history.metadata['timing']
                num_moves = len(game_history)
                print(f'cnt:{self.count}:Game played in {timing:.2f}s, {num_moves} moves ==> {timing / num_moves:.2f}s per move!')
                print(game_history)
            self.replay_buffer.save_history(game_history)


def tensorboard_model_summary(model: tf.keras.Model, line_length: int = 100) -> str:
    lines = []
    model.summary(print_fn=lambda line: lines.append(line), line_length=line_length)
    lines.insert(3, '-'*line_length)
    positions = [lines[2].find(col) for col in ['Layer', 'Output', 'Param', 'Connected']]
    positions.append(line_length)
    table = ['|'+'|'.join([line[positions[i]:positions[i+1]] for i in range(len(positions)-1)])+'|' for line in lines[2:-4] if line[0] not in ['=', '_']]
    result = '# Model summary\n' + '\n'.join(table) + '\n\n# Parameter summary\n' + '\n\n'.join(lines[-4:-1])
    return result


def play_train():

    # the models and the checkpoint file path
    db_base = 'F:/muzerottt2/'
    # the total trained epoch including history，if the cnt < 100 and wait_to_play=True, at the end of the epoch, the training process will wait 5 minutes for the playing
    start_epoch_cnt = 1000
    # 用于恢复模型的起点文件名，应根据最新的训练情况使用指定的起始点
    start_ckpt = 'ckpt-2'

    config = make_config()
    config.mcts_config.temperature = 4.35
    env = TicTacToeEnvironment()

    network = TicTacToeNetwork(config, config.network_config.network_parameters.get('regularizer'),
                               config.network_config.network_parameters.get('hidden_state_size'),
                               config.network_config.network_parameters.get('hidden_size'))
    print('representation')
    print(network.representation.input_shape)
    print(network.representation.output_shape)

    print('dynamics')
    print(network.dynamics.input_shape)
    print(network.dynamics.output_shape)

    print('prediction')
    print(network.prediction.input_shape)
    print(network.prediction.output_shape)

    print('state_preprocessing')
    print(network.state_preprocessing.input_shape)
    print(network.state_preprocessing.output_shape)

    print('recurrent_inference_model')
    print(network.recurrent_inference_model.input_shape)
    print(network.recurrent_inference_model.output_shape)

    print('initial_inference_model')
    print(network.initial_inference_model.input_shape)
    print(network.initial_inference_model.output_shape)

    env.reset()
    saved_models = db_base+'models'
    logdir = db_base+'logs'

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9)
    writer = tf.summary.create_file_writer(logdir) if logdir else None
    checkpoint = tf.train.Checkpoint(network=network.checkpoint, optimizer=optimizer)
    checkpoint_path = os.path.join(logdir, 'ckpt') if logdir else None
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=None) if logdir else None
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

    network.save_tfx_models(saved_models)
    if os.path.exists(db_base+'logs/ckpt/checkpoint'):
        checkpoint.read(db_base+'logs/ckpt/'+start_ckpt)

    replay_buffer = ReplayBuffer(config)
    wait_to_play = True
    if os.path.exists(db_base+'games.data'):
        replay_buffer.load_games(db_base+'games.data')
        wait_to_play = False

    agent = MCTSAgent(config=config, network=network, agent_id='F')
    model = ttt_model(config=config, network=network, optimizer=optimizer)

    print('start playing until to 3000 then to start train.')
    player = Playing(agent=agent, replay_buffer=replay_buffer, env=env)
    player.start()

    while wait_to_play and player.count < 300:
        time.sleep(30)
    print('start train')
    ttt_train_network(config=config, network=network, replay_buffer=replay_buffer, unrolled_model=model,
                      saved_models_path=saved_models, writer=writer, checkpoint_manager=checkpoint_manager,
                      wait_to_play=wait_to_play, epoch_cnt=start_epoch_cnt)
    player.train_finished = True
    replay_buffer.save_games(db_base+'games.data')
    print('train finished')


if __name__ == '__main__':
    play_train()
