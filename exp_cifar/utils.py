from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class RunningAverageEstimator:

    def __init__(self, gamma=.9):
        self.estimates = dict()
        self.gamma = gamma

    def update(self, key, val):
        if key in self.estimates.keys():
            self.estimates[key] = (self.gamma * self.estimates[key] +
                                   (1 - self.gamma) * val)
        else:
            self.estimates[key] = val

    def get(self, key):
        return self.estimates[key]


def get_binned_dataloaders(scores, dataloader, n_bins, n_examples):
    order = scores.argsort()
    binned = order.reshape(n_bins, -1)
    out_loaders = []
    source_tensors = dataloader.dataset.tensors
    for i in range(n_bins):
        indices = np.concatenate((binned[i][:n_examples // 2], binned[i][-n_examples // 2:]))
        tensors = (source_tensors[0][indices], source_tensors[1][indices], source_tensors[2][indices])
        dataset = TensorDataset(*tensors)
        out_loaders.append(DataLoader(dataset, batch_size=n_examples, shuffle=False))
    return out_loaders


def path_to_dict(path):
    return {k: v for k, v in [w.split('=') for w in path.split(',')]}