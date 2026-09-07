import random


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        self.q_table = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_q_values(self, state, legal_actions):
        if state not in self.q_table:
            self.q_table[state] = {}

        for action in legal_actions:
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0

        return self.q_table[state]

    def choose_action(self, state, legal_actions):

        q_values = self.get_q_values(state, legal_actions)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation
        max_q = max(q_values[action] for action in legal_actions)

        best_actions = [
            action
            for action in legal_actions
            if q_values[action] == max_q
        ]

        return random.choice(best_actions)

    def choose_greedy_action(self, state, legal_actions):

       q_values = self.q_table.get(state, {})

       max_q = max(
            q_values.get(action, 0.0)
            for action in legal_actions
        )

       best_actions = [
            action
            for action in legal_actions
            if q_values.get(action, 0.0) == max_q
        ]

       return random.choice(best_actions)

    
    
    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_legal_actions,
        done
    ):
        for transformed_state, action_map in self._symmetries(state):
            transformed_next_state = self._transform_state(next_state, action_map)
            transformed_action = action_map.index(action)
            transformed_next_actions = [
                action_map.index(next_action)
                for next_action in next_legal_actions
            ]
            self._update_once(
                transformed_state,
                transformed_action,
                reward,
                transformed_next_state,
                transformed_next_actions,
                done,
            )

    def _update_once(
        self,
        state,
        action,
        reward,
        next_state,
        next_legal_actions,
        done,
    ):

        q_values = self.get_q_values(state, [action])

        current_q = q_values[action]

        if done:
            target = reward

        else:
            next_q_values = self.get_q_values(
                next_state,
                next_legal_actions
            )

            best_next_q = max(
                next_q_values[action]
                for action in next_legal_actions
            )

            target = (
                reward
                + self.discount_factor * best_next_q
            )

        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        self.q_table[state][action] = new_q

    def _symmetries(self, state):
        if len(state) != 9:
            return [(state, tuple(range(len(state))))]

        mappings = (
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
            (6, 3, 0, 7, 4, 1, 8, 5, 2),
            (8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 5, 8, 1, 4, 7, 0, 3, 6),
            (2, 1, 0, 5, 4, 3, 8, 7, 6),
            (6, 7, 8, 3, 4, 5, 0, 1, 2),
            (0, 3, 6, 1, 4, 7, 2, 5, 8),
            (8, 5, 2, 7, 4, 1, 6, 3, 0),
        )
        return [
            (self._transform_state(state, mapping), mapping)
            for mapping in mappings
        ]

    @staticmethod
    def _transform_state(state, mapping):
        return tuple(state[index] for index in mapping)

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
            )
        
    def reset(self):

        self.q_table = {}
        self.epsilon = 1.0
