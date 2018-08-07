
class FixedQTarget:
    def __init__(self, target_model, target_update_step, use_soft_targets=False):
        self._target_model = target_model
        self._target_update_step = target_update_step
        self._use_soft_targets = use_soft_targets
        self._tau = 1.0 / self._target_update_step
        self._n_steps = 0

    def predict(self, states):
        return self._target_model.predict(states)

    def reset(self, online_model):
        self._target_model.set_weights(online_model.get_weights())
        self._n_steps = 0

    def step(self, online_model):
        if self._use_soft_targets:
            # Soft update fixed-Q targets
            weights_model = online_model.get_weights()
            weights_target = self._target_model.get_weights()
            new_weights = []

            for i in range(len(weights_model)):
                new_weights.append(self._tau * weights_model[i] + (1. - self._tau) * weights_target[i])

            self._target_model.set_weights(new_weights)
        else:
            if self._n_steps % self._target_update_step == 0:
                self._target_model.set_weights(online_model.get_weights())
