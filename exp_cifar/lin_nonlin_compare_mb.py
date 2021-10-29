# %%

import torch
import glob
import os
from nngeometry.object import PVector, PushForwardImplicit
from nngeometry.generator import Jacobian
import copy
from tasks import get_task
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from tasks import extract_small_loader
from margin_utils import margin_mean

path = './results/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=60,l2=0.0,lr=0.05,mom=0.0,seed=1,task=cifar10_resnet18,width=0'

ckpt_paths = sorted(glob.glob(os.path.join(path, 'checkpoint_*')))

# %%

n_examples = 250

# %%

with open('cifar10-cscores-orig-order.npz', 'rb') as f:
    d = np.load(f)
    cscores = d['scores']
    labels = d['labels']

# %% 

cscores = cscores[:n_examples]
labels = labels[:n_examples]
order = cscores.argsort()
ranks = order.argsort()

# %%

Args = namedtuple('Args', 'task batch_size depth width batch_norm')
args = Args(task='cifar10_resnet18', batch_size=125, depth=0, width=0, batch_norm=False)
_, dataloaders, criterion = get_task(args)
dataloaders['mini_train'] = extract_small_loader(dataloaders['train_deterministic'], n_examples, n_examples)


# %%

def analyze(loader, model, model_0, optimizer0, alpha=1):
    deltas_lin = []
    deltas_nonlin = []
    w_0 = PVector.from_model(model).clone().detach()

    for inputs, targets, logits in iter(loader):

        if len(deltas_lin) == 2:
            break

        optimizer = optimizer0
        optimizer.zero_grad()
        outputs = model(inputs)
        with torch.no_grad():
            outputs -= model_0(inputs).detach()
        outputs_alpha = alpha * outputs

        loss = criterion(outputs_alpha + logits, targets)
        grad_f = torch.autograd.grad(loss, outputs_alpha)[0]
        outputs.backward(grad_f)

        optimizer.step()
        optimizer.zero_grad()

        inputs, targets, logits = next(iter(dataloaders['mini_train']))
        outputs_after = model(inputs)
        with torch.no_grad():
            outputs_after -= model_0(inputs).detach()

        w_after = PVector.from_model(model).clone().detach()

        w_0.copy_to_model(model)

        outputs_before = model(inputs)
        with torch.no_grad():
            outputs_before -= model_0(inputs).detach()

        with torch.no_grad():
            deltas_nonlin.append(outputs_after - outputs_before)

        generator = Jacobian(model=model, n_output=10)
        pf = PushForwardImplicit(generator, examples=dataloaders['mini_train'])
        d_logits = pf.mv(w_after - w_0)

        deltas_lin.append(d_logits.get_flat_representation().t())
    
    return deltas_lin, deltas_nonlin
        

margin_scores = []
accs = []

for ckpt_path in ckpt_paths:

    print(ckpt_path)
    ckpt = torch.load(ckpt_path)
    accs.append(float(ckpt_path.split('/')[-1].split('_')[1]))

    deltas_lin, deltas_nonlin = analyze(dataloaders['train'], ckpt['model'], ckpt['model_0'],
                                        ckpt['optimizer'])

    delta_lin_sum = sum(deltas_lin)
    delta_nonlin_sum = sum(deltas_nonlin)

    diff = delta_nonlin_sum - delta_lin_sum

    margin_scores.append(margin_mean(diff.cpu(), labels))

    if False:
        plt.figure()
        plt.scatter(ranks, margin_scores[-1], alpha=.5)
        # plt.scatter(ranks, diff.cpu()[np.arange(n_examples), labels], alpha=.5)
        plt.title(ckpt_path.split('/')[-1])
        plt.grid()

        ylims = plt.ylim()
        ylim_max = max(-ylims[0], ylims[1])
        plt.ylim(-ylim_max, ylim_max)
        plt.show()

# %%

deltas_by_bins = []
for margin in margin_scores:
    ranked = margin[order]
    averaged = ranked.reshape(10, -1).mean(axis=1)

    plt.figure()
    plt.scatter(np.arange(10), averaged)
    plt.show()

    deltas_by_bins.append(averaged.numpy())

# %%

deltas_by_bins = np.array(deltas_by_bins)
# dbb = (deltas_by_bins - deltas_by_bins.mean(axis=1, keepdims=True)) / (deltas_by_bins.max(axis=1, keepdims=True) - deltas_by_bins.min(axis=1, keepdims=True))
# dbb = deltas_by_bins / np.abs(deltas_by_bins).max(axis=1, keepdims=True)
# dbb = deltas_by_bins / np.linalg.norm(deltas_by_bins, axis=1, keepdims=True)
dbb = deltas_by_bins
mm = max(-dbb.min(), dbb.max())
plt.imshow(dbb.T, cmap='PiYG', vmin=-mm, vmax=mm)
plt.ylabel('cscore bin')
cbar = plt.colorbar()
plt.xticks(np.linspace(0, len(accs)-1, 5), np.array(accs)[np.linspace(0, len(accs)-1, 5).astype('int')])
plt.xlabel('train accuracy')
cbar.ax.set_ylabel('delta', rotation=270)
# %%
