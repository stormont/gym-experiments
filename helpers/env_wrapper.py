
import numpy as np
import random


class EnvironmentWrapper:
    def __init__(self, env, n_bootstrap_steps=10000, verbose=1):
        self._env = env
        self._n_samples = 0
        self._mean = None
        self._std = None
        self._verbose = verbose

        if n_bootstrap_steps is not None:
            self._bootstrap(n_bootstrap_steps)

    def _bootstrap(self, n_bootstrap_steps):
        self._mean = None
        self._std = None
        steps = 0

        if self._verbose > 0:
            print('Bootstrapping environment stats over {} random time steps...'.format(n_bootstrap_steps))

        while steps < n_bootstrap_steps:
            done = False
            _ = self.reset()

            while not done:
                steps += 1
                action = random.randrange(self._env.action_space.n)
                _, _, done, _ = self.step(action)

        if self._verbose > 0:
            print('Bootstrapping complete; mean {}, std {}'.format(self._mean, self._std))

    def _update_env_stats(self, sample):
        # Incremental mean/standard deviation
        self._n_samples += 1

        if self._mean is None:
            self._std = np.repeat(1.0, len(sample))
            self._mean = sample
        else:
            self._std = (self._n_samples - 2) / (self._n_samples - 1) * self._std + \
                        (1 / self._n_samples) * np.square(sample - self._mean)
            self._mean += (sample - self._mean) / self._n_samples

    def _standardize(self, state):
        if self._mean is None or self._std is None:
            return state

        return (state - self._mean) / self._std

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def render(self):
        self._env.render()

    def reset(self):
        state = self._env.reset()
        self._update_env_stats(state)
        return self._standardize(state)

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        self._update_env_stats(state)
        return self._standardize(state), reward, done, info
