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
from plot_utils import *
from train_utils import makedir_lazy

if False:
    # lr=0.01, base epochs=197
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=197,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0'
    path_alpha1 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=196,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_alpha100 = os.path.join(base_exp, 'alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths = [path_alpha1, path_alpha100]
    exp_names = ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)']
elif False:
    # lr=0.01, base epochs=2
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=2,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0'
    path_alpha1 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_alpha100 = os.path.join(base_exp, 'alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths = [path_alpha1, path_alpha100]
    exp_names = ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)']
elif False:
    # lr=0.01, base epochs=3
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=3,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0'
    path_alpha1 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=250,fork=Truel2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_alpha100 = os.path.join(base_exp, 'alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=2000,fork=Truel2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths = [path_alpha1, path_alpha100]
    exp_names = ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)']
elif True:
    # lr=0.01, base epochs=3
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1,l2=0.0,lr=0.003,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0'
    path_alpha1 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_alpha100 = os.path.join(base_exp, 'alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=2000,fork=Truel2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths = [path_alpha1, path_alpha100]
    exp_names = ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)']
#
elif False:
    # lr=0.01, base epochs=1, vgg11
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,width=0/children/checkpoint_10_0'
    path_alpha1 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=250,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,track_lin=True,width=0')
    path_alpha100 = os.path.join(base_exp, 'alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_vgg11,track_accs=True,track_lin=True,width=0')
    paths = [path_alpha1, path_alpha100]
    exp_names = ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)']

elif False:
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1,l2=0.0,lr=0.003,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0'
    path_lr01 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_lr003 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.003,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_lr0001 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.0001,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_lr000001 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=10000,fork=True,l2=0.0,lr=1e-06,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths = [path_lr01, path_lr003, path_lr0001, path_lr000001]
    exp_names = ['lr=.01', 'lr=.003', f'lr=.0001', 'lr=1e-6']
elif False:
    base_exp = '/home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=2,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0'
    path_alpha1 = os.path.join(base_exp, 'alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=200,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    path_alpha100 = os.path.join(base_exp, 'alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0')
    paths = [path_alpha1, path_alpha100]
    exp_names = ['non-linear ($\\alpha=1$)', f'linear ($\\alpha=100$)']

# path = '~/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=197,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=196,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0'

# paths_fork = [
#     # '~/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=197,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0/alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=196,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0',
#     '~/projects/linvsnonlin/cifar10_noisy/alpha=1.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=197,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,width=0/children/checkpoint_10_0/alpha=100.0,batch_size=125,depth=0,diff=0.15,diff_type=random,epochs=1000,fork=True,l2=0.0,lr=0.01,mom=0.9,task=cifar10_resnet18,track_accs=True,track_lin=True,width=0',
# ]

figures_base_path = '/network/projects/g/georgeth/linvsnonlin/cifar10_noisy_figures'
figures_path = os.path.join(figures_base_path, base_exp.split('/')[8])
makedir_lazy(figures_path)

# d = pd.read_pickle(os.path.join(path, 'log.pkl'))

# %%

def get_checkpoint_name(path_fork):
    return path_fork.split('/')[6]

from plot_helpers import concatenate_acc_loss
# accs, losses = concatenate_acc_loss(d)

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
  gamma = .5
  o = []
  ra = x[0]
  for xi in x:
    ra = gamma * ra + (1 - gamma) * xi
    o.append(ra)
  return np.array(o)

no_smoothing = lambda x: x

smoothen = smoothen_running_average

# %%

def smoothen_xy_sorted(x, y):
    indices_sorted = x.argsort()

    return smoothen(x[indices_sorted]), smoothen(y[indices_sorted])

def smoothen_xy(x, y):
    return smoothen(x), smoothen(y)


# %% 

def plot_vs_train_loss(path, path_fork, normalize='none'):
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
        plt.xlabel(f'{normalize} c-score subgroup train loss')
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

    checkpoint_name = get_checkpoint_name(path_fork)
    plt.suptitle(checkpoint_name)
    cmap = cm.viridis(np.linspace(0, .8, n_bins))
    for i in range(n_bins):
        plt.scatter(*smoothen_xy(x, losses[:, i]), marker='x', color=cmap[i])
        plt.scatter(*smoothen_xy(x_fork, losses_fork[:, i]), marker='.', color='red')#cmap[i], alpha=.5)
    # plt.xscale('log')

    plt.ylabel('train loss - subsets ranked by cscore')
    # plt.ylim(0, 1)
    plt.grid()

    xlims = plt.xlim()
    ylims = plt.ylim()

    ## -- test

    plt.subplot2grid((2, 1), (1, 0))

    test_accs, test_losses = concatenate_acc_loss(d, train=False)
    test_accs_fork, test_losses_fork = concatenate_acc_loss(d_fork, train=False)
    
    # plt.scatter(x, d['test_loss'], label='regular', marker='x')
    # plt.scatter(x_fork, d_fork['test_loss'], label=f'alpha={fork_dict["alpha"]}', marker='+')
    for i in range(n_bins):
        plt.scatter(*smoothen_xy(x, test_losses[:, i]), marker='x', color=cmap[i])
        plt.scatter(*smoothen_xy(x_fork, test_losses_fork[:, i]), marker='.', color='red')#cmap[i], alpha=.5)
    plt.grid()
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.ylabel('test loss')
    
    plt.savefig(f'{figures_path}/{checkpoint_name}_alpha_{fork_dict["alpha"]}_loss_vs_{normalize}_bins.pdf')
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

import statsmodels.api as sm
from plot_helpers import smoothen_2d

# %%

def plot_noisy_vs_clean(paths, legends, test=True):

    f = create_figure(.5, 1.5)
    for path, legend in zip(paths, legends):
        # run_dict = path_to_dict(os.path.split(path)[-1])
        # print(run_dict)

        d = pd.read_pickle(os.path.join(path, 'log.pkl'))

        n_datapoints = len(d)
        x = d['train_easy_acc']
        y = d['train_diff_acc']
        y_test = d['test_acc']

        if False:
            plt.scatter(x, y,
                        label=legend,
                        marker='x')
        elif False:
            x_eval = np.arange(x.min(), x.max(), .01)
            smoothed = sm.nonparametric.lowess(exog=x, endog=y,
                                            frac=20/n_datapoints,
                                            it=0,
                                            xvals=x_eval)

            plt.plot(x_eval, smoothed, linewidth=3, label=legend)
        else:
            x_smooth, y_smooth = smoothen_2d(*smoothen_xy(x, y))
            print('subs', len(x_smooth), len(x))
            # plt.scatter(x_smooth, y_smooth,
            #             label=legend,
            #             marker='x')
            p = plt.plot(x_smooth, y_smooth, label=f'{legend} noisy', linewidth=1.5)

            if test:
                x_t_smooth, y_t_smooth = smoothen_2d(*smoothen_xy(x, y_test))
                plt.plot(x_t_smooth, y_t_smooth, '--',
                         linewidth=1.5, color=p[0].get_color(),
                         label=f'{legend} test')

    plt.xlabel('clean examples accuracy')
    plt.ylabel('accuracies')
    plt.grid()
    plt.legend()

    # plt.savefig(f'{figures_path}/acc_noisy_vs_clean.pdf')
    save_fig(f, f'{figures_path}/acc_noisy_vs_clean.pdf')
    plt.show()

plot_noisy_vs_clean(paths, exp_names)
# %%

from plot_helpers import plot_lin_vs_clean
plot_lin_vs_clean(paths, exp_names, figures_path=figures_path)

# %%

for path_fork in []:#paths_fork:
    for normalize in ['none', 'mean', 'low', 'middle', 'high']:
        # plot_vs_train_loss(path, path_fork, normalize=normalize)
        # plot_vs_train_acc(path, path_fork, normalize=normalize)
        # plot_vs_train_acc(path, path_fork, normalize=normalize, normalize_acc=False)
        pass

# %%
