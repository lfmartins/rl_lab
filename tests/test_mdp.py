# tests/test_mdp.py
import math
import pytest
from rl_lab import MDP

def test_public_api_import():
    """
    Sanity check: user can import MDP from the public API.
    """
    assert MDP is not None

recycling_robot_cases = [
    (0.8, 0.6, 15, 10, -3),
    (0.9, 0.3, 20, 12, -5),
    (0.4, 0.5, 12,  8, -1)
]
def test_recycling_robot_creation():
    states = ["high", "low"]
    actions = ["search", "wait", "recharge"]

    for alpha, beta, rsearch, rwait, rempty in recycling_robot_cases:
        transitions = {
            "high": {
                "search": [("high", alpha, rsearch), ("low", "*", rsearch)],
                "wait": [("high", "*", rwait)],
            },
            "low": {
                "search": [("low", beta, rsearch), ("high", "*", rempty)],
                "wait": [("low", "*", rwait)],
                "recharge": [("high", "*", 0.0)]
            }
        }

        mdp = MDP(states, actions, transitions)

        # States and actions
        assert set(mdp.states) == {"high", "low"}
        assert set(mdp.actions) == {"search", "wait", "recharge"}

        # Terminal states
        assert len(mdp.terminal_states) == 0

        # Admissible actions
        for state, actions in mdp.admissible_actions.items():
            for action in actions:
                assert action in mdp.actions

        # Transition probabilities sum to 1
        for state in mdp.states:
            for action in mdp.admissible_actions[state]:
                probs = mdp.P(state, action)
                assert math.isclose(probs.sum(), 1.0), f"P sum != 1 for {state}, {action}"

        # Rewards are numbers
        for state in mdp.states:
            for action in mdp.admissible_actions[state]:
                rewards = mdp.R(state, action)
                assert all(isinstance(r, float) for r in rewards), f"Rewards not floats for {state}, {action}"
