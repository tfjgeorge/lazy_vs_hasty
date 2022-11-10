# %%
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import copy

from plots import generate_example1

xs, ys, mus = generate_example1()

xs = torch.tensor(xs).float()
ys = torch.tensor(ys).float().t()

# %%

num_iterations = 50

# %%
class MLP(nn.Module):
    def __init__(self, param_scaling=1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(xs.shape[1], xs.shape[1], bias=False)
        self.linear2 = nn.Linear(xs.shape[1], 1, bias=False)

        with torch.no_grad():
            self.linear1.weight.mul_(param_scaling)
            self.linear2.weight.mul_(param_scaling)

    def forward(self, x):
        out = self.linear2(self.linear1(x))
        return out

model_0 = MLP()

# %%
criterion = nn.MSELoss(reduction='none')

def get_losses(lr, alpha, num_iterations, model):
    with torch.no_grad():
        pred_0 = model_0(xs)

    losses = []
    for it in range(num_iterations):
        pred = alpha * (model(xs) - pred_0)
        loss = criterion(pred, ys)
        losses.append(loss.detach())
        loss.mean().backward()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(p.grad, alpha=-lr / alpha**2)
                p.grad.zero_()
    return np.array([l.numpy() for l in losses])

# %%


lr = 1e-3

for model_i in range(5):
    model_0 = MLP(param_scaling=.5)

    for alpha, num_iterations in zip([.1, 10], [150, 4000]):
        model = copy.deepcopy(model_0)
        losses_np = get_losses(lr, alpha, num_iterations, model)
        plt.figure()
        plt.plot(losses_np[:, :, 0])
        plt.xlabel('GD iterations')
        plt.ylabel('per example squared loss')
        plt.savefig(f'plots_mlp/{model_i}_alpha_{alpha}.pdf')

# %%

    # %%
