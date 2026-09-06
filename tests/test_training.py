from agents.q_learning import QLearningAgent
from training.trainer import train


def test_training_creates_learning():

    agent = QLearningAgent()

    train(agent, 100)

    assert len(agent.q_table) > 0


def test_training_returns_metrics():

    agent = QLearningAgent()

    metrics = train(agent, 100)

    assert len(metrics.games) == 100
    assert len(metrics.rewards) == 100
    assert len(metrics.epsilon) == 100
    assert len(metrics.states) == 100