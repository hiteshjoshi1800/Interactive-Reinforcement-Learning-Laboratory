class TrainingMetrics:

    def __init__(self):
        self.games = []
        self.rewards = []
        self.epsilon = []
        self.states = []

    def record(
        self,
        game_number,
        reward,
        epsilon,
        number_of_states
    ):
        self.games.append(game_number)
        self.rewards.append(reward)
        self.epsilon.append(epsilon)
        self.states.append(number_of_states)

    def get_data(self):
        return {
            "games": self.games,
            "rewards": self.rewards,
            "epsilon": self.epsilon,
            "states": self.states
        }
    