
import numpy as np
import time


class DQNAgent:
    def __init__(self, env, model, gamma, exploration, experience=None, fixed_q_target=None, n_steps=1):
        self._env = env
        self._model = model
        self._gamma = gamma
        self._exploration = exploration
        self._experience = experience
        self._fixed_q_target = fixed_q_target
        self._n_steps = n_steps

        if self._fixed_q_target is not None:
            self._fixed_q_target.reset(self._model)

    @property
    def exploration(self):
        return self._exploration

    def _get_predictions(self, samples):
        states, actions, rewards, next_states, dones, sample_weights, sample_indices = samples
        predictions = np.zeros((len(states), self._model.action_size))

        action_returns = self._model.predict(states)
        next_action_returns = self._get_next_action_returns(next_states)
        sampled_returns = []

        for idx in range(len(states)):
            action, reward, done, action_return = actions[idx], rewards[idx], dones[idx], action_returns[idx]
            policy_action = self._select_policy_action(next_states, next_action_returns, idx)
            discounted_return = self._gamma * next_action_returns[idx][policy_action] * (not done)
            action_return[action] = reward + discounted_return
            predictions[idx] = action_return
            sampled_returns.append(action_return)

            if self._experience is not None and self._experience.supports_prioritization:
                importance, sample_idx = sample_weights[idx], sample_indices[idx]
                td_error = (action_return - self._model.predict(np.array([states[idx]])))[0][action]
                self._experience.update_priority(sample_idx, td_error)

        return predictions, sample_weights, sampled_returns

    def _get_next_action_returns(self, next_states):
        if self._fixed_q_target is not None:
            # Fixed-Q targets use next action returns from the target policy (off-policy)
            return self._fixed_q_target.predict(next_states)
        else:
            # Get the next action returns from the on-policy model
            return self._model.predict(next_states)

    def _sample_experience(self, state, action, reward, next_state, done):
        if self._experience is not None:
            # if self._experience.supports_prioritization:
            #     # Schaul 2015 says that incoming samples have no known error, but this seems incorrect; it seems
            #     # we can estimate the sample's TD error upon arrival.
            #     # Jaromir Janisch makes the same conclusion; see: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
            #     action_return = reward + self._get_next_action_returns(np.array([next_state]))
            #     td_error = (action_return - self._model.predict(np.array([state])))[0][action]
            # else:
            #     td_error = None

            self._experience.add(state, action, reward, next_state, done)
            return self._experience.sample()
        else:
            # This is a "vanilla" DQN
            return np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]),\
                   None, None

    def _select_policy_action(self, next_states, next_action_returns, sample_idx):
        if self._fixed_q_target is not None and self._fixed_q_target.use_double_q:
            # Double-Q selects the greedy action of the on-policy model (but evaluates it off-policy)
            return np.argmax(self._model.predict(next_states)[sample_idx])
        else:
            # Select the greedy action from the action returns given
            return np.argmax(next_action_returns[sample_idx])

    def _do_n_steps(self, action):
        # TODO: Complete redo this
        next_state = None
        reward = 0
        done = False
        step_action = action

        for i in range(self._n_steps):
            step_next_state, step_reward, step_done, _ = self._env.step(step_action)
            reward += step_reward * (self._gamma ** i)

            if next_state is None:
                # Just store the "first" next state when using multiple N-steps
                next_state = step_next_state

            if done is None:
                # Just store the "first" done when using multiple N-steps
                done = step_done

            if step_done:
                break

            # Get the greedy action using the on-policy model
            step_action = np.argmax(self._model.predict(np.array([next_state]))[0])

        return next_state, reward, done, dict()

    def train(self, render=False, debug_func=None):
        state = self._env.reset()
        total_reward = 0
        done = False
        n_steps = 0
        start_time = time.time()
        step_rewards = []
        losses = []

        while not done:
            if render:
                self._env.render()

            action = self._exploration.act(self._model, np.array([state]))
            # next_state, reward, done, _ = self._do_n_steps(action)
            next_state, reward, done, _ = self._env.step(action)
            samples = self._sample_experience(state, action, reward, next_state, done)

            if self._fixed_q_target is not None:
                self._fixed_q_target.step(self._model)

            states = samples[0]
            predictions, sample_weights, sampled_returns = self._get_predictions(samples)
            history = self._model.fit(states, predictions, sample_weight=sample_weights)
            losses.extend(history.history['loss'])

            # The Hasselt 2010 algorithm calls for randomly swapping models each update step.
            # This isn't seen much in DQN implementations, but seems more theoretically sound.
            if self._fixed_q_target is not None:
                self._model = self._fixed_q_target.swap_models(self._model)

            state = next_state
            total_reward += reward
            n_steps += 1
            step_rewards.extend(sampled_returns)

        if self._experience is not None:
            self._experience.step()

        self._exploration.step()

        # Allow the chance to examine the model for debugging
        if debug_func is not None:
            debug_func(self._model)

        elapsed_time = time.time() - start_time
        return total_reward, n_steps, elapsed_time, np.mean(step_rewards), np.mean(losses)

    def test(self, render=False, verbose=0):
        state = self._env.reset()
        total_reward = 0
        done = False
        n_steps = 0
        step_rewards = []

        while not done:
            if render:
                self._env.render()

            # Act greedily during testing
            action = np.argmax(self._model.predict(np.array([state]))[0])
            next_state, reward, done, _ = self._env.step(action)

            if verbose > 0:
                print('Step {}: reward {}, state {}, action {}'.format(n_steps, reward, state, action))

            state = next_state
            total_reward += reward
            n_steps += 1
            step_rewards.append(reward)

        return step_rewards, n_steps
