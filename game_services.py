import tensorflow as tf

from protos import training_runtime_pb2
from game import GameHistory
from utils import to_bytes_dict, from_bytes_dict


from muzero_types import Observation, Policy, Value, Action, Player, State


def history_to_protobuf(game_history: GameHistory) -> training_runtime_pb2.GameHistory:
    message = training_runtime_pb2.GameHistory()
    st_list = []
    for state in game_history.states:
        state_message = training_runtime_pb2.State()
        state_message.observation.CopyFrom(tf.make_tensor_proto(state.observation))
        state_message.legal_actions.extend(state.legal_actions)
        state_message.to_play = state.to_play
        st_list.append(state_message)
    message.states.extend(st_list)
    message.actions.extend(game_history.actions)
    message.rewards.extend(game_history.rewards)
    message.root_values.extend(game_history.root_values)
    message.policies.extend([tf.make_tensor_proto(policy) for policy in game_history.policies])
    message.metadata.update(to_bytes_dict(game_history.metadata))
    return message


def history_from_protobuf(message: training_runtime_pb2.GameHistory) -> GameHistory:
    history = GameHistory()
    for state in message.states:
        observation = Observation(tf.constant(tf.make_ndarray(state.observation)))
        legal_actions = [Action(action) for action in state.legal_actions]
        to_play = Player(state.to_play)
        history.states.append(State(observation, to_play, legal_actions))

    history.actions = [Action(index) for index in message.actions]
    history.rewards = [Value(reward) for reward in message.rewards]
    history.root_values = [Value(root_value) for root_value in message.root_values]
    history.policies = [Policy(tf.constant(tf.make_ndarray(policy))) for policy in message.policies]
    history.metadata.update(from_bytes_dict(message.metadata))

    return history
