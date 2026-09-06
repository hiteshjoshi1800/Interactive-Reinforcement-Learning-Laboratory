from environment.tictactoe import TicTacToe
from agents.q_learning import QLearningAgent
from agents.random_agent import RandomAgent



def evaluate(agent, number_of_games=1000):

    random_agent = RandomAgent()

    wins = 0
    draws = 0
    losses = 0

    for _ in range(number_of_games):

        game = TicTacToe()

        while not game.is_terminal():

            state = game.get_state()
            legal_actions = game.get_legal_actions()

            if game.current_player == 1:
                action = agent.choose_greedy_action(
                    state,
                    legal_actions
                )
            else:
                action = random_agent.choose_action(
                    state,
                    legal_actions
                )

            game.step(action)

        winner = game.check_winner()

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / number_of_games,
        "draw_rate": draws / number_of_games,
        "loss_rate": losses / number_of_games
    }