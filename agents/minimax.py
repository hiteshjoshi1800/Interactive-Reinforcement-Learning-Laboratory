from functools import lru_cache


class MinimaxAgent:
    """Perfect Tic-Tac-Toe opponent used to teach safe long-term play."""

    winning_lines = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    )

    def choose_action(self, state, legal_actions=None):
        if legal_actions is None:
            legal_actions = [index for index, value in enumerate(state) if value == 0]
        return min(
            legal_actions,
            key=lambda action: self._score(self._play(state, action, -1), True),
        )

    @lru_cache(maxsize=None)
    def _score(self, state, maximizing):
        winner = self._winner(state)
        if winner == 1:
            return 10
        if winner == -1:
            return -10
        legal_actions = [index for index, value in enumerate(state) if value == 0]
        if not legal_actions:
            return 0
        scores = [
            self._score(self._play(state, action, 1 if maximizing else -1), not maximizing)
            for action in legal_actions
        ]
        return max(scores) if maximizing else min(scores)

    @staticmethod
    def _play(state, action, player):
        board = list(state)
        board[action] = player
        return tuple(board)

    def _winner(self, state):
        for a, b, c in self.winning_lines:
            if state[a] and state[a] == state[b] == state[c]:
                return state[a]
        return 0
