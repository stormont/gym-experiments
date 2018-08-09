
from algorithms.schedule import ExponentialSchedule
import numpy as np
import random


class ExponentialEpsilonGreedyExploration:
    def __init__(self, epsilon_start, epsilon_min, epsilon_decay):
        self._schedule = ExponentialSchedule(start=epsilon_start, end=epsilon_min, step=epsilon_decay)

    @property
    def epsilon(self):
        return self._schedule.value

    def act(self, model, state):
        if np.random.rand() <= self._schedule.value:
            return random.randrange(model.action_size)

        # predict() returns a matrix tensor (even for a single state prediction),
        # but the action values are always a vector, so grab the first (and only) row
        return np.argmax(model.predict(state)[0])

    def step(self):
        self._schedule.step()
