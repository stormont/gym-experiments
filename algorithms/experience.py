
from collections import deque
import numpy as np
import random


class ExperienceReplay:
    def __init__(self, maxlen, sample_batch_size, min_size_to_sample):
        self._states = deque(maxlen=maxlen)
        self._actions = deque(maxlen=maxlen)
        self._rewards = deque(maxlen=maxlen)
        self._next_states = deque(maxlen=maxlen)
        self._dones = deque(maxlen=maxlen)
        self._sample_batch_size = sample_batch_size
        self._min_size_to_sample = min_size_to_sample

    def __len__(self):
        return len(self._states)

    @property
    def supports_prioritization(self):
        return False

    def add(self, state, action, reward, next_state, done, priority=None):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

    def bootstrap(self, env):
        print('Bootstrapping experience samples...')

        while not self.can_sample():
            state = env.reset()
            done = False

            while not done:
                action = np.random.randint(low=0, high=env.action_space.n)
                next_state, reward, done, _ = env.step(action)
                self.add(state, action, reward, next_state, done)

    def can_sample(self):
        return len(self) >= self._min_size_to_sample

    def sample(self):
        mem_size = len(self)
        indices = random.sample(range(mem_size), min(mem_size, self._sample_batch_size))
        states = np.array([self._states[idx] for idx in indices])
        actions = np.array([self._actions[idx] for idx in indices])
        rewards = np.array([self._rewards[idx] for idx in indices])
        next_states = np.array([self._next_states[idx] for idx in indices])
        dones = np.array([self._dones[idx] for idx in indices])
        return states, actions, rewards, next_states, dones, None, indices

    def step(self):
        # No-op
        pass

    def update_priority(self, idx, priority):
        # No-op
        pass


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, maxlen, sample_batch_size, min_size_to_sample, e=0.000001, alpha_sched=None, beta_sched=None):
        super(PrioritizedExperienceReplay, self).__init__(maxlen, sample_batch_size, min_size_to_sample)
        self._priorities = deque(maxlen=maxlen)
        self._e = e
        self._alpha_sched = alpha_sched
        self._beta_sched = beta_sched

    @property
    def supports_prioritization(self):
        return True

    def add(self, state, action, reward, next_state, done, priority=0.0):
        alpha = 1.0 if self._alpha_sched is None else self._alpha_sched.value
        priority = abs(priority) ** alpha + self._e

        if self.__len__() < self._states.maxlen:
            # Just append to the end
            self._states.append(state)
            self._actions.append(action)
            self._rewards.append(reward)
            self._next_states.append(next_state)
            self._dones.append(done)
            self._priorities.append(priority)
        else:
            # Replace the smallest existing priority
            min_idx = np.argmin(self._priorities)
            self._states[min_idx] = state
            self._actions[min_idx] = action
            self._rewards[min_idx] = reward
            self._next_states[min_idx] = next_state
            self._dones[min_idx] = done
            self._priorities[min_idx] = priority

    def sample(self):
        num_samples = self.__len__()
        dist = np.array(list(self._priorities))
        sum_priorities = dist.sum()
        beta = 1.0 if self._beta_sched is None else self._beta_sched.value
        dist /= sum_priorities  # Normalize distribution

        indices = np.random.choice(range(num_samples), self._sample_batch_size, p=dist)
        states = np.array([self._states[idx] for idx in indices])
        actions = np.array([self._actions[idx] for idx in indices])
        rewards = np.array([self._rewards[idx] for idx in indices])
        next_states = np.array([self._next_states[idx] for idx in indices])
        dones = np.array([self._dones[idx] for idx in indices])
        importances = np.array([(1. / (num_samples * dist[idx])) ** beta for idx in indices])

        return states, actions, rewards, next_states, dones, importances, indices

    def step(self):
        if self._alpha_sched is not None:
            self._alpha_sched.step()

        if self._beta_sched is not None:
            self._beta_sched.step()

    def update_priority(self, idx, priority):
        alpha = 1.0 if self._alpha_sched is None else self._alpha_sched.value
        self._priorities[idx] = abs(priority) ** alpha + self._e
