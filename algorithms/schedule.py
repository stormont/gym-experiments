
class ExponentialSchedule:
    def __init__(self, start, end, step):
        self._value = start
        self._end = end
        self._step = step

    @property
    def value(self):
        return self._value

    def step(self):
        # Simple exponential multiplication step on epsilon (until the end value is reached)
        if self._step < 1:
            self._value = max(self._value * self._step, self._end)
        else:
            self._value = min(self._value * self._step, self._end)


class LinearSchedule:
    def __init__(self, start, end, step):
        self._value = start
        self._end = end
        self._step = step

    @property
    def value(self):
        return self._value

    def step(self):
        # Simple linear change (until end value is met)
        if self._step < 0:
            self._value = max(self._value - self._step, self._end)
        else:
            self._value = min(self._value + self._step, self._end)
