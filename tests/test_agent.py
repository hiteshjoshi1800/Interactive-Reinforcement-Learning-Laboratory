from agents.q_learning import QLearningAgent


def test_q_table_starts_empty():
    agent = QLearningAgent()

    assert len(agent.q_table) == 0


def test_q_values_are_initialized():
    agent = QLearningAgent()

    state = (0, 0, 0)
    actions = [0, 1, 2]

    q_values = agent.get_q_values(state, actions)

    assert q_values[0] == 0.0
    assert q_values[1] == 0.0
    assert q_values[2] == 0.0


def test_q_value_changes_after_update():
    agent = QLearningAgent()

    state = (0, 0, 0)
    next_state = (1, 0, 0)

    agent.update(
        state,
        0,
        1,
        next_state,
        [1, 2],
        True
    )

    assert agent.q_table[state][0] != 0.0


def test_reset_clears_learning():
    agent = QLearningAgent()

    agent.q_table[(0, 0, 0)] = {
        0: 1.0
    }

    agent.epsilon = 0.05

    agent.reset()

    assert len(agent.q_table) == 0
    assert agent.epsilon == 1.0