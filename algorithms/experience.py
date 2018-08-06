
from collections import deque
import numpy as np
import random


class ExperienceReplay:
    def __init__(self, maxlen, sample_batch_size, min_size_to_sample):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.sample_batch_size = sample_batch_size
        self.min_size_to_sample = min_size_to_sample

    def __len__(self):
        return len(self.states)

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

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
        return len(self) >= self.min_size_to_sample

    def sample(self):
        mem_size = len(self)
        indices = random.sample(range(mem_size), min(mem_size, self.sample_batch_size))
        states = np.array([self.states[idx] for idx in indices])
        actions = [self.actions[idx] for idx in indices]
        rewards = [self.rewards[idx] for idx in indices]
        next_states = np.array([self.next_states[idx] for idx in indices])
        dones = [self.dones[idx] for idx in indices]
        return states, actions, rewards, next_states, dones
