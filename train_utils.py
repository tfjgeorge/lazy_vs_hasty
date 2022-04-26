import os

class Recorder():
    def __init__(self):
        self.values = dict()

    def save(self, key, val, i=None):
        if i is not None:
            val = (i, val)
        if key in self.values.keys():
            self.values[key].append(val)
        else:
            self.values[key] = [val]

    def get(self, key):
        return self.values[key]

    def len(self, key):
        return len(self.values[key])

def makedir_lazy(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ProbeAssistant:

    def __init__(self, init_loss, reduce_factor, gamma=.66):
        self._loss = init_loss
        self._reduce_factor = reduce_factor
        self._gamma = gamma
        self._next_threshold = init_loss * reduce_factor
        self._probe = 0

    def record_loss(self, loss):
        self._loss = self._loss * self._gamma + loss * (1 - self._gamma)
        if self._loss <= self._next_threshold:
            self._probe += 1
            self._next_threshold = self._reduce_factor * self._next_threshold

    def do_probe(self):
        if self._probe > 0:
            self._probe -= 1
            return True
        return False
