# %%

from matplotlib.colors import Normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from utils import path_to_dict
import os
import sys
sys.path.append('..')
from plot_utils import create_figure, save_fig

path = '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0'

paths_fork = [
    # '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0',
    '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0',
    '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_30_320/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0',
    '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_50_960/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0',
    '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_70_2560/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0',
    '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=198,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_90_4800/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0',
    # '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=199,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_20_384/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0',
    # '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=199,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_40_1920/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0',
    # '~/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=200,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_20_384/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0'
]
# paths_fork = [
#     'results/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=113,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_20_392/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0',
#     'results/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=113,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_50_3136/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0',
#     'results/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=113,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_75_8624/alpha=10000.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.02,mom=0.0,task=cifar10_resnet18,track_accs=True,width=0',
# ]

if False:
    #resnet18
    path = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0'

    paths_fork = [
        '/home/mila/g/georgeth/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0/alpha=100.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0'
    ]
elif False:
    # vgg11
    path = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,width=0/children/checkpoint_10_0/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,track_lin=True,width=0'

    paths_fork = [
        '/home/mila/g/georgeth/projects/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=1,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,width=0/children/checkpoint_10_0/alpha=100.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=25000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,track_lin=True,width=0'
    ]
elif True:
    base_path = '/network/scratch/g/georgeth/linvsnonlin/cifar10/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=3,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/'
    path = os.path.join(base_path, 'children/checkpoint_0_0/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths_fork = [
        os.path.join(base_path, 'children/checkpoint_0_0/alpha=100.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=2500,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    ]

figures_path = '/network/scratch/g/georgeth/linvsnonlin/cifar10_figures'

d = pd.read_pickle(os.path.join(path, 'log.pkl'))

# %%

def get_checkpoint_name(path_fork):
    return path_fork.split('/')[10]

# %% 
from plot_helpers import concatenate_acc_loss

accs, losses = concatenate_acc_loss(d)

def concatenate(d, col_name):
    cc = []
    for i, r in d.iterrows():
        cc.append(r[col_name])

    cc = np.array(cc)

    return cc

cummin = np.minimum.accumulate

# %%

from scipy.ndimage.filters import gaussian_filter1d

smoothen_gaussian = lambda x: gaussian_filter1d(x, sigma=2)

N = 5
# smoothen_moving_average = lambda x: np.convolve(x, np.ones(N)/N, mode='valid')
def smoothen_moving_average(x):
    x = np.array(x)
    n_expand = N // 2
    x_expanded = np.concatenate(([x[0]]*n_expand, x, [x[-1]]*n_expand))
    return np.convolve(x_expanded, np.ones(N)/N, mode='valid')

def smoothen_running_average(x):
  gamma = .9
  o = []
  ra = x[0]
  for xi in x:
    ra = gamma * ra + (1 - gamma) * xi
    o.append(ra)
  return np.array(o)

no_smoothing = lambda x: x

smoothen = smoothen_moving_average

# %%

def smoothen_xy(x, y):
    indices_sorted = x.argsort()

    return smoothen(x[indices_sorted]), smoothen(y[indices_sorted])

from plot_utils import smoothen_interpolate as smoothen_xy

smoothen_xy = lambda *x: x

# %% 
linewidth = .7
def plot_vs_train_loss(path, path_fork, normalize='none'):
    fork_dict = path_to_dict(os.path.split(path_fork)[-1])
    print(fork_dict)

    d = pd.read_pickle(os.path.join(path, 'log.pkl'))
    d_fork = pd.read_pickle(os.path.join(path_fork, 'log.pkl'))

    print(int(d_fork['iteration'].max() / 400), d_fork['time'].max())

    accs, losses = concatenate_acc_loss(d)
    accs_fork, losses_fork = concatenate_acc_loss(d_fork)

    n_bins = losses.shape[1]

    create_figure(.5, 1.5)
    # plt.subplot2grid((2, 1), (0, 0))

    if normalize in ['mean', 'low', 'middle', 'high']:
        if normalize == 'mean':
            x = losses.mean(axis=1)
            x_fork = losses_fork.mean(axis=1)
        elif normalize == 'low':
            x = losses[:, 0]
            x_fork = losses_fork[:, 0]
        elif normalize == 'middle':
            x = losses[:, n_bins // 2]
            x_fork = losses_fork[:, n_bins // 2]
        elif normalize == 'high':
            x = losses[:, -1]
            x_fork = losses_fork[:, -1]
        plt.xlabel(f'{normalize} training loss')
        # plt.xlim(d_fork['train_loss'].max(), 0)
        plt.xlim(x_fork.max()*1.1, x_fork.min()*.7)
        if False:
            xtent = d_fork['train_loss'].max() - d_fork['train_loss'].min()
            plt.xlim(d_fork['train_loss'].max() + xtent * .2, d_fork['train_loss'].min() - xtent * .2)
    else:
        x = d['iteration']
        x_fork = d_fork['iteration']
        plt.xlabel('sgd iterations')
        if True:
            xtent = d['iteration'].max()
            plt.xlim(0, xtent*1.2)

    _x = np.minimum.accumulate(x)
    _x_fork = np.minimum.accumulate(x_fork)
    # _x = x
    # _x_fork = x_fork

    checkpoint_name = get_checkpoint_name(path_fork)
    cmap = cm.viridis(np.linspace(0, .8, n_bins))
    for i in range(n_bins):
        plt.plot(*smoothen_xy(_x, losses[:, i]),
                 marker='', color=cmap[i], linewidth=linewidth)
        plt.plot(*smoothen_xy(_x_fork, losses_fork[:, i]), '--',
                 marker='', color=cmap[i], linewidth=linewidth)

    plt.plot([], [], '', color='black', linewidth=linewidth,
             label='non-linear ($\\alpha=1$)')
    plt.plot([], [], '--', color='black', linewidth=linewidth,
             label='linear ($\\alpha=100$)')
    plt.plot([], [], '', color=cmap[0], linewidth=2,
             label='lowest c-scores')
    plt.plot([], [], '', color=cmap[-1], linewidth=2,
             label='highest c-scores')
    plt.legend()
    # plt.xscale('log')

    plt.ylabel('per c-score bin training loss')
    # plt.ylim(0, 1)
    plt.grid()

    xlims = plt.xlim()
    ylims = plt.ylim()

    ## -- test

    # plt.subplot2grid((2, 1), (1, 0))

    # test_accs, test_losses = concatenate_acc_loss(d, train=False)
    # test_accs_fork, test_losses_fork = concatenate_acc_loss(d_fork, train=False)
    
    # # plt.scatter(x, d['test_loss'], label='regular', marker='x')
    # # plt.scatter(x_fork, d_fork['test_loss'], label=f'alpha={fork_dict["alpha"]}', marker='+')
    # for i in range(n_bins):
    #     plt.plot(*smoothen_xy(x, test_losses[:, i]),
    #              marker='', color=cmap[i], linewidth=linewidth)
    #     # plt.plot(*smoothen_xy(x_fork, test_losses_fork[:, i]),
    #     #          marker='', color='red')#cmap[i], alpha=.5)
    #     plt.plot(*smoothen_xy(x_fork, test_losses_fork[:, i]), '--',
    #              marker='', color=cmap[i], linewidth=linewidth)
    # plt.grid()
    # plt.xlim(*xlims)
    # plt.ylim(*ylims)
    # plt.ylabel('test loss')
    
    fig_path = f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_loss_vs_{normalize}_bins.pdf'
    save_fig(plt.gcf(), fig_path)
    plt.show()

def plot_vs_train_acc(path, path_fork, normalize='none', normalize_acc=True):
    fork_dict = path_to_dict(os.path.split(path_fork)[-1])
    print(fork_dict)

    d = pd.read_pickle(os.path.join(path, 'log.pkl'))
    d_fork = pd.read_pickle(os.path.join(path_fork, 'log.pkl'))

    print(int(d_fork['iteration'].max() / 400), d_fork['time'].max())

    accs, losses = concatenate_acc_loss(d)
    accs_fork, losses_fork = concatenate_acc_loss(d_fork)

    n_bins = losses.shape[1]

    plt.figure(figsize=(8, 8))
    plt.subplot2grid((2, 1), (0, 0))

    suptitle_add = ''
    if normalize in ['mean', 'low', 'middle', 'high']:
        if normalize_acc:
            norm = accs
            norm_fork = accs_fork
            norm_str = 'acc'
        else:
            norm = losses
            norm_fork = losses_fork
            norm_str = 'loss'
        if normalize == 'mean':
            x = norm.mean(axis=1)
            x_fork = norm_fork.mean(axis=1)
            suptitle_add = f'normalized by mean train {norm_str}'
        elif normalize == 'low':
            x = norm[:, 0]
            x_fork = norm_fork[:, 0]
            suptitle_add = f'normalized by train {norm_str} of low cscore subgroup'
        elif normalize == 'middle':
            x = norm[:, n_bins // 2]
            x_fork = norm_fork[:, n_bins // 2]
            suptitle_add = f'normalized by train {norm_str} of middle cscore subgroup'
        elif normalize == 'high':
            x = norm[:, -1]
            x_fork = norm_fork[:, -1]
            suptitle_add = f'normalized by train {norm_str} of high cscore subgroup'
    else:
        x = d['iteration']
        x_fork = d_fork['iteration']
        plt.xlabel('sgd iterations')
        xtent = d_fork['iteration'].max()
        plt.xlim(0, xtent*1.2)
        norm_str = ''

    checkpoint_name = get_checkpoint_name(path_fork)
    plt.suptitle(checkpoint_name + '\n' + suptitle_add)
    cmap = cm.viridis(np.linspace(0, .8, n_bins))
    for i in range(n_bins):
        plt.scatter(*smoothen_xy(x, accs[:, i]), marker='x', color=cmap[i])
        plt.scatter(*smoothen_xy(x_fork, accs_fork[:, i]), marker='.', color='red')
    # plt.xscale('log')
    # plt.xlim(2.5, 1.5)
    plt.ylabel('train acc - subsets ranked by cscore')
    # plt.ylim(0, 1)
    plt.grid()
    if normalize != 'none' and normalize_acc:
        xlims = plt.xlim()
    else:
        xlims = plt.xlim()
        xlims = (xlims[1], xlims[0])
        plt.xlim(xlims)

    ## -- test

    plt.subplot2grid((2, 1), (1, 0))

    test_accs, test_losses = concatenate_acc_loss(d, train=False)
    test_accs_fork, test_losses_fork = concatenate_acc_loss(d_fork, train=False)

    for i in range(n_bins):
        plt.scatter(*smoothen_xy(x, test_accs[:, i]), marker='x', color=cmap[i])
        plt.scatter(*smoothen_xy(x_fork, test_accs_fork[:, i]), marker='.', color='red')
    plt.grid()
    plt.xlim(*xlims)
    plt.ylabel('test accuracy')

    plt.savefig(f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_acc_vs_{normalize}_{norm_str}_bins.pdf')
    plt.show()

# %%

from scipy.interpolate import interp1d

def plot_diff_vs_train_loss(path, path_fork, normalize='none'):
    fork_dict = path_to_dict(os.path.split(path_fork)[-1])
    print(fork_dict)

    d = pd.read_pickle(os.path.join(path, 'log.pkl'))
    d_fork = pd.read_pickle(os.path.join(path_fork, 'log.pkl'))

    print(int(d_fork['iteration'].max() / 400), d_fork['time'].max())

    accs, losses = concatenate_acc_loss(d)
    accs_fork, losses_fork = concatenate_acc_loss(d_fork)

    n_bins = losses.shape[1]

    create_figure(.5, 1.5)
    # plt.subplot2grid((2, 1), (0, 0))

    if normalize in ['mean', 'low', 'middle', 'high']:
        if normalize == 'mean':
            x = losses.mean(axis=1)
            x_fork = losses_fork.mean(axis=1)
        elif normalize == 'low':
            x = losses[:, 0]
            x_fork = losses_fork[:, 0]
        elif normalize == 'middle':
            x = losses[:, n_bins // 2]
            x_fork = losses_fork[:, n_bins // 2]
        elif normalize == 'high':
            x = losses[:, -1]
            x_fork = losses_fork[:, -1]
        plt.xlabel(f'{normalize} training loss')
        # plt.xlim(d_fork['train_loss'].max(), 0)
        plt.xlim(x_fork.max()*1.1, x_fork.min()*.7)
    else:
        x = d['iteration']
        x_fork = d_fork['iteration']
        plt.xlabel('sgd iterations')
        if True:
            xtent = d['iteration'].max()
            plt.xlim(0, xtent*1.2)

    _x = np.minimum.accumulate(x)
    _x_fork = np.minimum.accumulate(x_fork)

    x_common = np.linspace(min(_x.min(), _x_fork.min()),
                           max(_x.max(), _x_fork.max()), 200)

    checkpoint_name = get_checkpoint_name(path_fork)
    cmap = cm.viridis(np.linspace(0, .8, n_bins))
    for i in range(n_bins):
        f = interp1d(_x, losses[:, i], bounds_error=False)
        f_fork = interp1d(_x_fork, losses_fork[:, i], bounds_error=False)
        y = f(x_common)
        y_fork = f_fork(x_common)
        plt.plot(x_common, y - y_fork, marker='', color=cmap[i],
                 linewidth=linewidth)

    plt.plot([], [], '', color=cmap[0], linewidth=2,
             label='lowest c-scores')
    plt.plot([], [], '', color=cmap[-1], linewidth=2,
             label='highest c-scores')
    plt.legend()
    # plt.xscale('log')

    plt.ylabel('nonlinear - linear training loss')
    # plt.ylim(0, 1)
    plt.grid()

    plt.xlim(np.log(10), 0)

    fig_path = f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_loss_diff_vs_{normalize}_bins.pdf'
    save_fig(plt.gcf(), fig_path)
    plt.show()

# %%

def plot_diff_vs_train_acc(path, path_fork, normalize='none', test=True):
    fork_dict = path_to_dict(os.path.split(path_fork)[-1])
    print(fork_dict)

    d = pd.read_pickle(os.path.join(path, 'log.pkl'))
    d_fork = pd.read_pickle(os.path.join(path_fork, 'log.pkl'))

    print(int(d_fork['iteration'].max() / 400), d_fork['time'].max())

    accs, losses = concatenate_acc_loss(d)
    accs_fork, losses_fork = concatenate_acc_loss(d_fork)

    n_bins = losses.shape[1]

    if test:
        create_figure(.5, 0.75)
        plt.subplot2grid((2, 1), (0, 0))
    else:
        create_figure(.5, 1.5)

    if normalize in ['mean', 'low', 'middle', 'high']:
        if normalize == 'mean':
            x = accs.mean(axis=1)
            x_fork = accs_fork.mean(axis=1)
        elif normalize == 'low':
            x = accs[:, 0]
            x_fork = accs_fork[:, 0]
        elif normalize == 'middle':
            x = accs[:, n_bins // 2]
            x_fork = accs_fork[:, n_bins // 2]
        elif normalize == 'high':
            x = accs[:, -1]
            x_fork = accs_fork[:, -1]

    _x = np.maximum.accumulate(x)
    _x_fork = np.maximum.accumulate(x_fork)

    x_common = np.linspace(min(_x.min(), _x_fork.min()),
                           max(_x.max(), _x_fork.max()), 200)

    checkpoint_name = get_checkpoint_name(path_fork)
    cmap = cm.viridis(np.linspace(0, .8, n_bins))
    for i in range(n_bins):
        f = interp1d(_x, accs[:, i], bounds_error=False)
        f_fork = interp1d(_x_fork, accs_fork[:, i], bounds_error=False)
        y = f(x_common)
        y_fork = f_fork(x_common)
        plt.plot(x_common, y - y_fork, marker='', color=cmap[i],
                 linewidth=linewidth)

    plt.plot([], [], '', color=cmap[0], linewidth=2,
             label='lowest c-scores')
    plt.plot([], [], '', color=cmap[-1], linewidth=2,
             label='highest c-scores')
    plt.legend()
    # plt.xscale('log')

    plt.ylabel('nonlinear - linear training accuracy')
    # plt.ylim(0, 1)
    plt.grid()

    plt.xlim(0, 1)

    if test:
        plt.subplot2grid((2, 1), (1, 0))

        test_accs, test_losses = concatenate_acc_loss(d, train=False)
        test_accs_fork, test_losses_fork = concatenate_acc_loss(d_fork, train=False)

        for i in range(n_bins):
            f = interp1d(_x, test_accs[:, i], bounds_error=False)
            f_fork = interp1d(_x_fork, test_accs_fork[:, i], bounds_error=False)
            y = f(x_common)
            y_fork = f_fork(x_common)
            plt.plot(x_common, y - y_fork, marker='', color=cmap[i],
                    linewidth=linewidth)

        plt.plot([], [], '', color=cmap[0], linewidth=2,
                label='lowest c-scores')
        plt.plot([], [], '', color=cmap[-1], linewidth=2,
                label='highest c-scores')
        plt.legend()
        # plt.xscale('log')

        plt.ylabel('nonlinear - linear test accuracy')
        # plt.ylim(0, 1)
        plt.grid()

        plt.xlim(0, 1)

    plt.xlabel(f'{normalize} training accuracy')

    fig_path = f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_acc_diff_vs_{normalize}_bins.pdf'
    save_fig(plt.gcf(), fig_path)
    plt.show()

# %%

for path_fork in paths_fork:
    for normalize in ['mean']:#['none', 'mean', 'low', 'middle', 'high']:
        plot_vs_train_loss(path, path_fork, normalize=normalize)
        plot_diff_vs_train_loss(path, path_fork, normalize=normalize)
        plot_vs_train_acc(path, path_fork, normalize=normalize)
        plot_diff_vs_train_acc(path, path_fork, normalize=normalize)
        # plot_vs_train_acc(path, path_fork, normalize=normalize, normalize_acc=False)


# %%

def smoothen_xy(x, y):
    return smoothen_running_average(x), smoothen_running_average(y)


def plot_lin_vs_clean(paths, legends, figures_path,
                      easy_fn=lambda d: d['train_easy_acc']):

    checkpoint_name = get_checkpoint_name(paths[1])
    fork_dict = path_to_dict(os.path.split(paths[1])[-1])

    f = create_figure(.5, ratio=1.5)

    plt.plot([], [], label='sign similarity', linewidth=1.5, color='black')
    plt.plot([], [], '--', label='repr. alignment', linewidth=1.5, color='black')
    plt.plot([], [], 'x-', label='ntk alignment', linewidth=1.5, color='black')

    for path, legend in zip(paths, legends):
        # run_dict = path_to_dict(os.path.split(path)[-1])
        # print(run_dict)

        d = pd.read_pickle(os.path.join(path, 'log.pkl'))

        n_datapoints = len(d)
        x = easy_fn(d)
        y_sign = d['sign_similarity']
        y_repr = d['repr_alignment']
        y_ntk = d['ntk_alignment']

        x_sign_s, y_sign_s = smoothen_xy(x,y_sign)
        p = plt.plot(x_sign_s, y_sign_s, label=legend, linewidth=1.5)

        x_repr_s, y_repr_s = smoothen_xy(x, y_repr)
        plt.plot(x_repr_s, y_repr_s, '--',
                    linewidth=1.5, color=p[0].get_color())

        x_ntk_s, y_ntk_s = smoothen_xy(x, y_ntk)
        plt.plot(x_ntk_s, y_ntk_s, 'x-',
                    linewidth=1.5, color=p[0].get_color())

    plt.xlabel('mean training loss')
    plt.ylabel('linearity metrics')
    plt.grid()
    plt.legend(ncol=2)
    xlims = plt.xlim()
    plt.xlim(xlims[1], xlims[0])

    save_fig(f, f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_lin_vs_loss.pdf')
    plt.show()

plot_lin_vs_clean([path, path_fork],
                  ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)'],
                  figures_path=figures_path,
                  easy_fn=lambda d: d['train_loss'])
# %%
