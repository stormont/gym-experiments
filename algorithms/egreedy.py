
import numpy as np
import random


class ExponentialEpsilonGreedyExploration:
    def __init__(self, epsilon_start, epsilon_min, epsilon_decay):
        self._epsilon = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    @property
    def epsilon(self):
        return self._epsilon

    def act(self, model, state):
        if np.random.rand() <= self._epsilon:
            return random.randrange(model.action_size)

        # predict() returns a matrix tensor (even for a single state prediction),
        # but the action values are always a vector, so grab the first (and only) row
        return np.argmax(model.predict(state)[0])

    def step(self):
        # Simple exponential decay on epsilon (until the minimum value is reached)
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)
