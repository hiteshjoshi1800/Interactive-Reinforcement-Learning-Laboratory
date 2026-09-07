import random


class RuleBasedAgent:
    """A lightweight opponent that wins, blocks, and prefers strong squares."""

    winning_lines = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    )

    def choose_action(self, state, legal_actions=None):
        if legal_actions is None:
            legal_actions = [index for index, value in enumerate(state) if value == 0]

        for action in legal_actions:
            if self._would_win(state, action, -1):
                return action

        for action in legal_actions:
            if self._would_win(state, action, 1):
                return action

        if 4 in legal_actions:
            return 4

        corners = [action for action in (0, 2, 6, 8) if action in legal_actions]
        if corners:
            return random.choice(corners)

        return random.choice(legal_actions)

    def _would_win(self, state, action, player):
        board = list(state)
        board[action] = player
        return any(all(board[index] == player for index in line) for line in self.winning_lines)
