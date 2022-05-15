# For type annotations
from typing import List, Dict, Optional, Any

from muzero_types import State, Observation, Player, Action, Value, Policy, ValueBatch, PolicyBatch, ActionBatch
from environment import Environment


class GameHistory(object):
    """
    Book-keeping class for completed games.
    Restricted to 1-player games with no intermediate rewards for MuProver.
    """

    def __init__(self) -> None:
        # self.observations: List[Observation] = []
        # self.to_plays: List[Player] = []
        self.states: List[State] = []
        self.actions: List[Action] = []
        self.rewards: List[Value] = []
        self.root_values: List[Value] = []
        self.policies: List[Policy] = []
        self.metadata: Dict[str, Any] = {}

        # If this history is from Reanalyse Buffer
        self.reanalyse: bool = False

        # The following are only filled once within a replay buffer
        self.extended_actions: Optional[ActionBatch] = None
        self.target_rewards: Optional[ValueBatch] = None
        self.target_values: Optional[ValueBatch] = None
        self.target_policies: Optional[PolicyBatch] = None
        self.total_value: Value = Value(float('nan'))

    def make_image(self, index: int = -1) -> Observation:
        """
        TODO: If necessary, stack multiple states to create an observation.
        """
        return self.states[index].observation

    def __repr__(self) -> str:
        return 'Game({})'.format(', '.join(map(str, self.actions)))

    def __len__(self) -> int:
        return len(self.actions)

    def is_reanalyse(self) -> bool:
        return self.reanalyse


class Game:
    """
    A class to record episodes of interaction with an Environment.
    """

    def __init__(self, environment: Optional[Environment]) -> None:
        self.environment: Environment = environment
        self.history: GameHistory = GameHistory()

        if environment:
            self.state: State = self.environment.reset()
            self.history.states.append(self.state)

        self.ended: bool = False
        self.debug: bool = False

    def to_play(self) -> Player:
        return self.state.to_play

    def legal_actions(self) -> List[Action]:
        return self.state.legal_actions

    def terminal(self) -> bool:
        return self.ended

    def apply(self, action: Action) -> None:
        if self.environment:
            self.state, reward, self.ended, info = self.environment.step(action)
            self.history.states.append(self.state)
            self.history.actions.append(action)
            self.history.rewards.append(reward)
            # print(self.history.observations)
            # print(f'players: {self.history.to_plays}')
            # print(f'reward: {self.history.rewards}')
        if self.debug:
            print(self.environment)

    def store_search_statistics(self, value: Value, policy: Policy) -> None:
        self.history.root_values.append(value)
        # print(f'root value:{value}')
        self.history.policies.append(policy)
        # print(f'policy: {policy}')


# class StoredGame(Game):
#     """ A stored Game that can be used for reanalyse ."""
#     def __init__(self, game: Game):
#         super().__init__(game.action_space_size, game.discount, environment=None)
#         self._stored_history = game.history
#
#     def terminal(self) -> bool:
#         return not self._stored_history.actions
#
#     def apply(self, action: Action):
#         # Ignore the action , instead replay the stored data .
#         del action
#         self.history.observations.append(self._stored_history.observations.pop(0))
#         self.history.to_plays.append(self._stored_history.to_plays.pop(0))
#         self.history.rewards.append(self._stored_history.rewards.pop(0))
#         self.history.actions.append(self._stored_history.actions.pop(0))
