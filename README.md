# MuProver

Tensorflow implementation of the MuZero algorithm, based on the pseudo-code provided
in the original paper:

**[1]** J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, 
E Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, D. Silver,
["Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"](https://arxiv.org/abs/1911.08265)

## Design

This implementation isolates the various components of MuZero, and uses
 gRPC for communication between them. This should make it straightforward
to deploy the algorithm in the cloud and scale the resources up to the point
required for solving complex problems.

The main components are:

- An environment server (`environment`).

- A replay buffer server (`replay`), storing the self-played games and producing training 
batches from these.

- A network server (`network`), performing the neural network evaluations required during
self-play (provided by TensorFlow Serving).

- A training agent (`training`), using the self-played games from `replay` to train the 
  neural networks in `network`.

- A Monte-Carlo Tree-Search agent (`agent`), playing games using the latest networks 
available in `network` to produce games for `replay`.
  
Remove the docker dependency(If you are building a large servers version, you can choose the main version from MuProver), 
so you can test it very simply!

## Installation

### Requirements

#### software versions

- tensorflow 2.7
- python 3.8 


#### Install `conda`

### Installing muzero

Clone this git repository and install required dependencies 
(**TODO: streamline installation**).

## Usage

Follow these steps to train MuZero to play a given game:

1. Start training the tictactoe
   First you have to config the training veriables:
   - db_base: the models and the checkpoint file path
   - start_epoch_cnt: the total trained epoch including history，if the cnt < 100 and wait_to_play=True, at the end of the epoch, the training process will wait 5 minutes for the playing
   
   ```
   python muzerottt.py
   ```
   
1. You can evaluate the model that you trained

   ```
   python ttt_evaluate.py
   ```

1. You can play with the computer

   ```
   python ttt_play_with_human.py
   ```

## Currently implemented games:

The following games have already been implemented (though only partial experiments 
have been carried out with them):

- TicTacToe (`tictactoe/tictactoe.py`).

#### Implementing other games

To implement a new game, you should sub-class the `Environment` class 
defined in `environment.py`, see `games/random_tictactoe.py` for an example. In 
the `games/yourgame.py` file you should also sub-class the `Network` class 
defined in `network.py` to define the neural networks used by MuProver for
your game. Finally, you should also provide a `make_config` method returning a
`MuZeroConfig` object (defined in `config.py`), containing all the
configuration parameters required by MuProver.

Alternatively, you may altogether skip creating an `Environment` sub-class
and simply define an environment server communicating through gRPC following
`protos/environment.proto`. If you do create the `Environment` sub-class, however, 
you will immediately be able to serve your environment using the standard server 
in `environment_services.py`.

## Custom training loops

You can define a custom training loop _e.g._ for synchronous training, 
whereby the same process alternates between self-playing games and training 
the neural networks. To do this, you may simply use the `Environment`,
`ReplayBuffer` and `Network` classes directly, instead of through their
`RemoteEnvironment`, `RemoteReplayBuffer` and `RemoteNetwork` counterparts.

However, you should be aware that this is certainly going to be much slower
than using the distributed, asynchroneous training.

## Notes

- You may want to tinker with `models/batching.config` and/or manually compile the 
TensorFlow Serving server to optimize network throughput in the target system.
