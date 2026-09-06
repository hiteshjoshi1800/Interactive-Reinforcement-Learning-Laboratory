class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self):
        # 0 = empty
        # 1 = X
        # -1 = O
        self.board = [0] * 9

        # X always starts
        self.current_player = 1

        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def get_legal_actions(self):
        return [
            i for i, cell in enumerate(self.board)
            if cell == 0
        ]

    def step(self, action):
        # Check if move is legal
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move")

        # Place current player's mark
        self.board[action] = self.current_player

        # Check if game is over
        winner = self.check_winner()

        if winner != 0:
            return self.get_state(), winner, True

        # Check for draw
        if len(self.get_legal_actions()) == 0:
            return self.get_state(), 0, True

        # Switch player
        self.current_player *= -1

        return self.get_state(), 0, False

    def check_winner(self):

        winning_combinations = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),

            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),

            (0, 4, 8),
            (2, 4, 6)
        ]

        for a, b, c in winning_combinations:

            if (
                self.board[a] != 0
                and self.board[a] == self.board[b]
                and self.board[b] == self.board[c]
            ):
                return self.board[a]

        return 0

    def is_terminal(self):
        return (
            self.check_winner() != 0
            or len(self.get_legal_actions()) == 0
        )