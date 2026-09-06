import random


class RandomAgent:

    def choose_action(self, state, legal_actions=None):
        if legal_actions is None:
            legal_actions = state
        return random.choice(legal_actions)