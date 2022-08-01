# %%

from collections import namedtuple
from matplotlib.colors import Normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from torch import threshold
from utils import path_to_dict
from tasks import get_task
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append('..')
from plot_utils import create_figure, save_fig

if True:
    base_epochs = 3
    base_path = f'/network/scratch/g/georgeth/linvsnonlin/svhn/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs={base_epochs},l2=0.0,lr=0.01,mom=0.9,task=svhn_resnet18,width=0/'
    path = os.path.join(base_path, 'children/checkpoint_0_0/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=svhn_resnet18,track_all_accs=True,track_lin=True,width=0')
    paths_fork = [
        os.path.join(base_path, 'children/checkpoint_0_0/alpha=100.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=2500,fork=True,l2=0.0,lr=0.01,mom=0.9,task=svhn_resnet18,track_all_accs=True,track_lin=True,width=0')
    ]

figures_path = '/network/scratch/g/georgeth/linvsnonlin/svhn_figures'

# %%

Args = namedtuple('Args', 'task batch_size depth width batch_norm')
args = Args(task='svhn_resnet18', batch_size=1000, depth=0, width=0, batch_norm=False)
_, dataloaders, criterion, criterion_noreduce = get_task(args)


# %%
def get_checkpoint_name(path_fork):
    return path_fork.split('/')[10]

# %% 
linewidth = .7
n_imgs = 8
train_ds = dataloaders['train'].dataset

digit_indices = []
for digit in range(0, 10):
    digit_indices.append(np.nonzero(train_ds[:1000][1] == digit))

print([(i, len(d)) for i, d in zip(range(10), digit_indices)])


# %%

def best_test_digits(path):
    d = pd.read_pickle(os.path.join(path, 'log.pkl'))

    best_it = np.argmin([l.mean() for l in d['test_all_losses']])
    print(best_it)

    losses = []
    for digit in range(10):
        loss = d['test_all_losses'][best_it][digit_indices[digit]].float().mean().item()
        losses.append(loss)
        print(digit, loss)
    return losses

best_test_losses_nonlin = best_test_digits(path)
plt.bar(range(10), best_test_losses_nonlin)
plt.show()

best_test_losses_lin = best_test_digits(paths_fork[0])
plt.bar(range(10), best_test_losses_lin)
plt.show()

# %%

def plot_vs_train_loss(path, path_fork, threshold, order=np.arange(10)):
    fork_dict = path_to_dict(os.path.split(path_fork)[-1])
    print(fork_dict)
    checkpoint_name = get_checkpoint_name(path_fork)

    d = pd.read_pickle(os.path.join(path, 'log.pkl'))
    d_fork = pd.read_pickle(os.path.join(path_fork, 'log.pkl'))

    # print(int(d_fork['iteration'].max() / 400), d_fork['time'].max())
    print(d.columns)
    # print(d_fork.columns)
    plt.plot(d['train_loss'], marker='+', label='nonlin')
    
    plt.plot(d_fork['train_loss'], marker='+',label='lin')

    plt.legend()

    fig_path = f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_loss.pdf'

    plt.gcf().patch.set_facecolor('grey')
    save_fig(plt.gcf(), fig_path)
    plt.show()

    it = np.where(d['train_loss'] < threshold)[0].min()
    it_fork = np.where(d_fork['train_loss'] < threshold)[0].min()
    print(it, it_fork)

    loss_diff = d['train_all_losses'][it] - d_fork['train_all_losses'][it_fork]

    loss_order = np.argsort(loss_diff)

    fig = plt.figure()
    gs = gridspec.GridSpec(n_imgs // 2, 2 * 2)
    for i in range(n_imgs):
        axis = fig.add_subplot(gs[4 * (i // 2) + i % 2])
        axis.imshow(train_ds[loss_order[i]][0].cpu().permute(1, 2, 0))
        axis = fig.add_subplot(gs[4 * (i // 2) + i % 2 + 2])
        axis.imshow(train_ds[loss_order[999-i]][0].cpu().permute(1, 2, 0))
    
    plt.show()

    accs = []
    accs_fork = []

    plt.figure()
    for digit in range(10):
        acc = d['train_all_accs'][it][digit_indices[digit]].float().mean().item()
        acc_fork = d_fork['train_all_accs'][it_fork][digit_indices[digit]].float().mean().item()
        accs.append(acc)
        accs_fork.append(acc_fork)
        print(digit, acc, acc_fork)
    plt.show()

    accs = np.array(accs)
    accs_fork = np.array(accs_fork)

    plt.bar(np.arange(10), (accs - accs_fork)[order])
    plt.show()

    losses = []
    losses_fork = []

    plt.figure()
    for digit in range(10):
        loss = d['train_all_losses'][it][digit_indices[digit]].float().mean().item()
        loss_fork = d_fork['train_all_losses'][it_fork][digit_indices[digit]].float().mean().item()
        losses.append(loss)
        losses_fork.append(loss_fork)
        print(digit, loss, loss_fork)
    plt.show()

    losses = np.array(losses)
    losses_fork = np.array(losses_fork)

    plt.bar(np.arange(10), (losses - losses_fork)[order])
    plt.show()


thresholds = [2, 1.5, 1., .5]
for thresh in thresholds:
    for path_fork in paths_fork:
        plot_vs_train_loss(path, path_fork, thresh)

# %%

thresholds = [2, 1.5, 1., .5]
for thresh in thresholds:
    for path_fork in paths_fork:
        plot_vs_train_loss(path, path_fork, thresh, order=np.argsort(best_test_losses_lin))

# %%
