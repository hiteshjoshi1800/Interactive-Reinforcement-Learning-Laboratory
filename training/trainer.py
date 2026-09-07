from environment.tictactoe import TicTacToe
from agents.q_learning import QLearningAgent
from agents.minimax import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from training.metrics import TrainingMetrics


def choose_opponent(agent, game_number):
    opponent_type = game_number % 10
    if opponent_type < 3:
        return RandomAgent()
    if opponent_type < 6:
        return RuleBasedAgent()
    if opponent_type < 8:
        return MinimaxAgent()
    return agent


def train_game(agent, game_number=0):

    game = TicTacToe()
    opponent = choose_opponent(agent, game_number)

    final_reward = 0

    while not game.is_terminal():
        if game.current_player == 1:
            state = game.get_state()
            action = agent.choose_action(state, game.get_legal_actions())
            next_state, reward, done = game.step(action)

            if done:
                agent.update(
                    state, action, reward, next_state, [], True
                )
                final_reward = reward
                continue

            opponent_action = (
                opponent.choose_action(
                    game.get_state(), game.get_legal_actions()
                )
                if opponent is not agent
                else opponent.choose_action(
                    game.get_state(), game.get_legal_actions()
                )
            )
            next_state, opponent_reward, done = game.step(opponent_action)
            agent.update(
                state,
                action,
                -1 if done and opponent_reward == -1 else 0,
                next_state,
                game.get_legal_actions(),
                done,
            )
            final_reward = -1 if done and opponent_reward == -1 else 0
        else:
            opponent_action = opponent.choose_action(
                game.get_state(), game.get_legal_actions()
            )
            _, reward, done = game.step(opponent_action)
            final_reward = reward

    agent.decay_epsilon()

    return final_reward


def train(agent, number_of_games):

    metrics = TrainingMetrics()

    for game_number in range(number_of_games):

        reward = train_game(agent, game_number)

        metrics.record(
            game_number=game_number + 1,
            reward=reward,
            epsilon=agent.epsilon,
            number_of_states=len(agent.q_table)
        )

        if (game_number + 1) % 1000 == 0:

            print(
                f"Games: {game_number + 1} | "
                f"States: {len(agent.q_table)} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return metrics


if __name__ == "__main__":

    agent = QLearningAgent()

    metrics = train(
        agent,
        10000
    )

    print("\nTraining complete!")
    print(
        "Total states learned:",
        len(agent.q_table)
    )

    print(
        "Total games:",
        len(metrics.games)
    )