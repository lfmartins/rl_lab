import numpy as np

from typing import (
    Hashable, Dict, List, Tuple, Union,
    Iterable, AbstractSet, Mapping,
)
from types import MappingProxyType

State = Hashable
Action = Hashable

Probability = Union[float, str]   # '*' allowed
Reward = float

Transition = Tuple[State, Probability, Reward]
ActionTransitions = Dict[Action, List[Transition]]
Transitions = Dict[State, ActionTransitions]


class MDP:
    """
    Minimal representation of a Markov decision process.
    """

    def __init__(self,
                 states: List[State],
                 actions: List[Action],
                 transitions: Transitions,
                 terminal_states: Iterable[State] = (),
                 seed=None) -> None:
        self._states = tuple(states)
        self._n_states = len(states)
        self._actions = tuple(actions)
        self._n_actions = len(actions)
        self._terminal_states = frozenset(terminal_states)
        self._admissible_actions = {}

        for state in self._terminal_states:
            self._admissible_actions[state]  = set()

        self._state_indexes = {}
        for index, state in enumerate(states):
            if state in self._state_indexes:
                raise ValueError(f"Repeated state in parameter states: {state}")
            self._state_indexes[state] = index

        self._action_indexes = {}
        for index, action in enumerate(actions):
            if action in self._action_indexes:
                raise ValueError(f"Repeated action in parameter actions: {action}")
            self._action_indexes[action] = index

        self._transition_probs = np.zeros(
            shape=(self._n_actions, self._n_states, self._n_states),
            dtype=np.float64)

        self._rewards = np.zeros(
            shape=(self._n_actions, self._n_states, self._n_states),
            dtype=np.float64)

        all_states = set(self._state_indexes)
        non_terminal_states = all_states - self._terminal_states
        transition_states = set(transitions)
        missing = non_terminal_states - transition_states
        if missing:
            raise ValueError(f"Missing transition specification for non-terminal "
                              "states: {missing}")

        tol = 1e-12
        for state, actions_dict in transitions.items():
            if state not in self._state_indexes:
                raise ValueError(f"Invalid state {state} in transitions.")
            if state in self._terminal_states:
                raise ValueError(f"Terminal state {state} cannot have transitions")

            j = self._state_indexes[state]
            self._admissible_actions[state] = set()

            for action, transition_list in actions_dict.items():
                if action not in self._action_indexes:
                    raise ValueError(f"Key {action} not valid for state {state} "
                                      "in parameter transitions.")
                if action in self._admissible_actions[state]:
                    raise ValueError(f"Action {action} is repeated for key {state} "
                                      "in parameter transitions.")
                self._admissible_actions[state].add(action)
                i = self._action_indexes[action]

                last_transition = False
                sum_probs = 0.0
                seen_targets = set()
                for target_state, p, r in transition_list:
                    if last_transition:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"No transitions allowed after a transition has probability '*'")
                    if target_state not in self._states:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"invalid state, {target_state}")
                    if target_state in seen_targets:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"state {target_state} repeated")

                    seen_targets.add(target_state)
                    k = self._state_indexes[target_state]

                    if p == '*':
                        p = 1 - sum_probs
                        if p < -tol:
                            raise ValueError(f"In transition list for state {state}, action {action}: "
                                              "probabilities add to more than 1")
                        last_transition = True
                    elif not isinstance(p, (int, float)):
                        raise TypeError(f"In transition list fr state {state}, action {action}: "
                                        f"transition probability to state {target_state} must be an number or '*'")
                    sum_probs += p

                    self._transition_probs[i, j, k] = p
                    self._rewards[i, j, k] = r

                if not last_transition:
                    raise ValueError(f"In transition list for state {state}, action {action}: "
                                      "probability for last transition must be '*'")

                if abs(sum_probs - 1.0) > tol:
                    raise ValueError(f"In transition list for state {state}, action {action}: "
                                      "transition probabilities do not add to 1")

        self._admissible_actions = {
            s: frozenset(a)
            for s, a in self._admissible_actions.items()
        }

        # Simulation set up
        self._rng = np.random.default_rng(seed=seed)
        self._current_state = None
        self._current_state_index = None
        self._terminated = True

        self._cumulative_probs = np.cumsum(self._transition_probs, axis=2)
        self._cumulative_probs[:, :, -1] = 1.0


    @property
    def states(self) -> Tuple[State, ...]:
        return self._states

    @property
    def actions(self) -> Tuple[Action, ...]:
        return self._actions

    @property
    def state_indexes(self) -> Mapping[State, int]:
        return MappingProxyType(self._state_indexes)

    @property
    def action_indexes(self) -> Mapping[Action, int]:
        return MappingProxyType(self._action_indexes)

    @property
    def terminal_states(self) -> AbstractSet[State]:
        return self._terminal_states

    def is_terminal(self, state: State) -> bool:
        return state in self._terminal_states

    @property
    def admissible_actions(self) -> Mapping[State, frozenset[Action]]:
        return MappingProxyType(self._admissible_actions)

    @property
    def transition_probs(self) -> np.ndarray:
        view = self._transition_probs.view()
        view.setflags(write=False)
        return view

    @property
    def rewards(self) -> np.ndarray:
        view = self._rewards.view()
        view.setflags(write=False)
        return view

    def seed(self, seed):
        self._rng = np.random.default_rng(seed=seed)

    def reset(self, state=None):
        if state is not None:
            if state not in self._states:
                raise ValueError(f"State {state} not valid")
            self._current_state = state
        else:
            self._current_state = self._rng.choice(self._states)
        self._current_state_index = self._state_indexes[self._current_state]
        self._terminated = False
        reward = 0.0
        return self._current_state, reward, self._terminated

    def step(self, action):
        if (self._current_state is None
            or self._current_state_index is None
            or self._terminated):
            raise ValueError("Simulation not set up properly. Call reset() before calling step()")
        if action not in self._admissible_actions[self._current_state]:
            raise ValueError(f"Action {action} not available for state {self._current_state}")
        action_index = self._action_indexes(action)
        u = self._rng.random()
        next_state_index = int(
            np.searchsorted(
                self._cumulative_probs[action_index, self._current_state_index],
                u,
                side="right"
            )
        )
        reward = self._rewards[action_index, self._current_state_index, next_state_index]
        self._current_state_index = next_state_index
        self._current_state = self._states[self._current_state_index]
        if self._current_state in self._terminal_states:
            self._terminated = True
        return self._current_state, reward, self._terminated







