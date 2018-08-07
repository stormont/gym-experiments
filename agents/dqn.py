
import numpy as np
import time


class DQNAgent:
    def __init__(self, env, model, gamma, exploration, experience=None, fixed_q_target=None):
        self._env = env
        self._model = model
        self._gamma = gamma
        self._exploration = exploration
        self._experience = experience
        self._fixed_q_target = fixed_q_target

        if self._fixed_q_target is not None:
            self._fixed_q_target.reset(self._model)

    @property
    def exploration(self):
        return self._exploration

    def _get_predictions(self, samples):
        states, actions, rewards, next_states, dones = samples
        predictions = np.zeros((len(states), self._model.action_size))

        action_returns = self._model.predict(states)

        if self._fixed_q_target is not None:
            next_action_returns = self._fixed_q_target.predict(next_states)
        else:
            next_action_returns = self._model.predict(next_states)

        for idx in range(len(states)):
            action, reward, done, action_return = actions[idx], rewards[idx], dones[idx], action_returns[idx]

            # argmax_a q_hat(S', a, w')
            if self._fixed_q_target is not None and self._fixed_q_target.use_double_q:
                greedy_action = np.argmax(self._model.predict(next_states)[idx])
            else:
                greedy_action = np.argmax(next_action_returns[idx])

            # gamma * q_hat(S', greedy_action, w')
            discounted_return = self._gamma * next_action_returns[idx][greedy_action] * (not done)
            # R + gamma * q_hat(S', argmax_a q_hat(S', a, w), w')
            action_return[action] = reward + discounted_return

            predictions[idx] = action_return

        return predictions

    def _sample_experience(self):
        return self._experience.sample()

    def _vanilla_dqn(self, state, action, reward, next_state, done):
        return np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done])

    def train(self, render=False):
        state = self._env.reset()
        total_reward = 0
        done = False
        n_steps = 0
        start_time = time.time()

        while not done:
            if render:
                self._env.render()

            action = self._exploration.act(self._model, np.array([state]))
            next_state, reward, done, _ = self._env.step(action)

            if self._experience is not None:
                self._experience.add(state, action, reward, next_state, done)
                samples = self._sample_experience()
            else:
                samples = self._vanilla_dqn(state, action, reward, next_state, done)

            if self._fixed_q_target is not None:
                self._fixed_q_target.step(self._model)

            predictions = self._get_predictions(samples)
            self._model.fit(samples[0], predictions)  # The first element of the `samples` tuple is `states`

            state = next_state
            total_reward += reward
            n_steps += 1

        self._exploration.step()

        elapsed_time = time.time() - start_time
        return total_reward, n_steps, elapsed_time
