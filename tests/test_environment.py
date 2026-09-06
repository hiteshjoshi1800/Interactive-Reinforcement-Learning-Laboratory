from environment.tictactoe import TicTacToe
import pytest


def test_board_starts_empty():
    game = TicTacToe()

    assert game.get_state() == (0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_all_moves_are_legal_at_start():
    game = TicTacToe()

    assert game.get_legal_actions() == list(range(9))


def test_move_is_applied():
    game = TicTacToe()

    game.step(4)

    assert game.board[4] == 1


def test_illegal_move_raises_error():
    game = TicTacToe()

    game.step(0)

    with pytest.raises(ValueError):
        game.step(0)


def test_reset_clears_board():
    game = TicTacToe()

    game.step(0)
    game.reset()

    assert game.get_state() == (0, 0, 0, 0, 0, 0, 0, 0, 0)