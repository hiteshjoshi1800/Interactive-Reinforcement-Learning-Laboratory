from environment.tictactoe import TicTacToe
from agents.q_learning import QLearningAgent
from training.metrics import TrainingMetrics


def train_game(agent):

    game = TicTacToe()

    final_reward = 0

    while not game.is_terminal():

        state = game.get_state()
        legal_actions = game.get_legal_actions()

        action = agent.choose_action(
            state,
            legal_actions
        )

        next_state, reward, done = game.step(action)

        next_legal_actions = game.get_legal_actions()

        agent.update(
            state,
            action,
            reward,
            next_state,
            next_legal_actions,
            done
        )

        final_reward = reward

    agent.decay_epsilon()

    return final_reward


def train(agent, number_of_games):

    metrics = TrainingMetrics()

    for game_number in range(number_of_games):

        reward = train_game(agent)

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