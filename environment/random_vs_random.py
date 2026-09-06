from tictactoe import TicTacToe
from random_agent import RandomAgent


def play_game():

    game = TicTacToe()

    x_agent = RandomAgent()
    o_agent = RandomAgent()

    while not game.is_terminal():

        if game.current_player == 1:
            agent = x_agent
        else:
            agent = o_agent

        legal_actions = game.get_legal_actions()

        action = agent.choose_action(legal_actions)

        state, reward, done = game.step(action)

    return reward


def run_games(number_of_games):

    x_wins = 0
    o_wins = 0
    draws = 0

    for _ in range(number_of_games):

        result = play_game()

        if result == 1:
            x_wins += 1

        elif result == -1:
            o_wins += 1

        else:
            draws += 1

    print(f"Games: {number_of_games}")
    print(f"X wins: {x_wins}")
    print(f"O wins: {o_wins}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    run_games(10000)