from agents.q_learning import QLearningAgent
from training.trainer import train
from evaluation.evaluator import evaluate
from persistence.model_manager import ModelManager


def run_experiment(
    number_of_games=10000,
    number_of_evaluation_games=1000
):
    # 1. Create a completely fresh agent
    agent = QLearningAgent()

    # 2. Train from zero
    metrics = train(
        agent,
        number_of_games
    )

    # 3. Evaluate the trained agent
    evaluation = evaluate(
        agent,
        number_of_evaluation_games
    )

    # 4. Collect experiment results
    run_data = {
        "training_games": number_of_games,
        "evaluation_games": number_of_evaluation_games,

        "final_states": len(agent.q_table),
        "final_epsilon": agent.epsilon,

        "wins": evaluation["wins"],
        "draws": evaluation["draws"],
        "losses": evaluation["losses"],

        "win_rate": evaluation["win_rate"],
        "draw_rate": evaluation["draw_rate"],
        "loss_rate": evaluation["loss_rate"]
    }

    # 5. Save experiment history
    model_manager = ModelManager()
    model_manager.save_run(run_data)

    # 6. Return results
    return run_data


if __name__ == "__main__":

    result = run_experiment(
        number_of_games=10000,
        number_of_evaluation_games=1000
    )

    print("\nExperiment complete!")
    print(result)