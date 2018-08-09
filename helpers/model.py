
class ModelWrapper:
    def __init__(self, model, fit_batch_size=32):
        self._model = model
        self._fit_batch_size = fit_batch_size

    @property
    def action_size(self):
        return self._model.layers[-1].output_shape[1]

    def fit(self, states, predictions, sample_weight=None):
        return self._model.fit(states, predictions, epochs=1, verbose=0, batch_size=self._fit_batch_size,
                               sample_weight=sample_weight)

    def get_weights(self):
        return self._model.get_weights()

    def predict(self, state):
        return self._model.predict(state)

    def set_weights(self, weights):
        self._model.set_weights(weights)
