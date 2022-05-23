import os
from torch.utils.data import DataLoader


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
        if key not in self.values.keys():
            return 0
        return len(self.values[key])

def makedir_lazy(path):
    if not os.path.exists(path):
        os.makedirs(path)


class ProbeAssistant:

    def __init__(self, callback, init_loss, reduce_factor, gamma=.66):
        self._loss = init_loss
        self._reduce_factor = reduce_factor
        self._gamma = gamma
        self._next_threshold = init_loss * reduce_factor
        self._probe = 0
        self._probe_callback = callback

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

    def step(self, force_probe=False):
        if self.do_probe() or force_probe:
            self._probe_callback()


def create_fname(d, exclude=[], replace=dict()):
    name = ''
    for k, v in sorted(d.items(), key=lambda a: a[0]):
        if k in exclude:
            continue
        if v is not False:
            if k in replace.keys():
                if v is not None:
                    name += replace[k] + ','
            else:
                name += '%s=%s,' % (k, str(v))
    return name[:-1]


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()
        self._i = -1
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
            self._epoch += 1
        self._i += 1
        return self._i, self._epoch, batch