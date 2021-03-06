
import numpy as np


class FixedQTarget:
    def __init__(self, target_model, target_update_step, use_soft_targets=False, use_double_q=False):
        self._target_model = target_model
        self._target_update_step = target_update_step
        self._use_soft_targets = use_soft_targets
        self._tau = 1.0 / self._target_update_step
        self._n_steps = 0
        self._use_double_q = use_double_q

    @property
    def use_double_q(self):
        return self._use_double_q

    def predict(self, states):
        return self._target_model.predict(states)

    def reset(self, policy_model):
        self._target_model.set_weights(policy_model.get_weights())
        self._n_steps = 0

    def step(self, policy_model):
        if self._use_soft_targets:
            # Soft update fixed-Q targets
            weights_model = policy_model.get_weights()
            weights_target = self._target_model.get_weights()
            new_weights = []

            for i in range(len(weights_model)):
                new_weights.append(self._tau * weights_model[i] + (1. - self._tau) * weights_target[i])

            self._target_model.set_weights(new_weights)
        else:
            if self._n_steps % self._target_update_step == 0:
                self._target_model.set_weights(policy_model.get_weights())

    def swap_models(self, policy_model):
        # To reduce/eliminate maximization bias, the target and policy networks should be randomly swapped
        # See Hasselt 2010: https://papers.nips.cc/paper/3964-double-q-learning.pdf
        if self._use_double_q and np.random.random() < 0.5:
            swapped_model = self._target_model
            self._target_model = policy_model
            return swapped_model
        else:
            return policy_model
