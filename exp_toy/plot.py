# %%
import numpy as np
import pickle as pkl
from viz_utils import custom_imshow
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from plot_utils import *
import torch

# %%

def filter(a, f, i):
    o = []
    for x in a:
        if len(x[f]) > i:
            o.append(x[f][i])
    return o
# %%

from datasets import (generate_yinyang_dataset, generate_twosquares_dataset,
                      generate_disk_dataset)


scratch_path = '/network/scratch/g/georgeth/'
base_path = os.path.join(scratch_path, 'linvsnonlin/toy/')
device = 'cpu'
test_res = 250

fig_path = os.path.join(scratch_path, 'linvsnonlin/toy_figures/')

to_plot = [
    # (os.path.join(base_path, 'dataset=yinyang1,extra_dims=0,loss=ce,max_iterations=5000,n_runs=100.pkl'),
    (os.path.join(base_path, 'dataset=yinyang1,extra_dims=0,loss=ce,loss_milestones=[0.5, 0.4, 0.3],max_its=25000,min_its=5000,n_runs=100.pkl'),
     lambda: generate_yinyang_dataset(n_train=1, p=0,
                                      test_resolution=test_res, variant=1,
                                      device=device)),
    (os.path.join(base_path, 'dataset=yinyang1,extra_dims=5,loss=ce,loss_milestones=[0.5, 0.4, 0.3],max_its=25000,min_its=5000,n_runs=100.pkl'),
     lambda: generate_yinyang_dataset(n_train=1, p=0,
                                      test_resolution=test_res, variant=1,
                                      device=device)),
    # (os.path.join(base_path, 'dataset=yinyang1,extra_dims=5,loss=ce,max_iterations=5000,n_runs=100.pkl'),
    #  lambda: generate_yinyang_dataset(n_train=1, p=0,
    #                                   test_resolution=test_res, variant=1,
    #                                   device=device)),
    # (os.path.join(base_path, 'two_squares_5_ce_20.pkl'),
    #  lambda: generate_twosquares_dataset(n_train=1, p=0,
    #                                      test_resolution=test_res,
    #                                      device=device)),
    # (os.path.join(base_path, 'dataset=disk_flip_vertical,extra_dims=0,loss=ce,max_iterations=5000,n_runs=100.pkl'),
    #  lambda: generate_disk_dataset(n_train=1, p=0,
    #                                test_resolution=test_res,
    #                                flip_half='vertical',
    #                                device=device)),
    (os.path.join(base_path, 'dataset=disk_flip_diagonal,extra_dims=0,loss=ce,loss_milestones=[0.5, 0.4, 0.3],max_its=25000,min_its=5000,n_runs=80.pkl'),
     lambda: generate_disk_dataset(n_train=1, p=0,
                                   test_resolution=test_res,
                                   flip_half='diagonal',
                                   device=device)),
    (os.path.join(base_path, 'dataset=disk_flip_diagonal,extra_dims=5,loss=ce,loss_milestones=[0.5, 0.4, 0.3],max_its=25000,min_its=5000,n_runs=80.pkl'),
     lambda: generate_disk_dataset(n_train=1, p=0,
                                   test_resolution=test_res,
                                   flip_half='diagonal',
                                   device=device)),
    (os.path.join(base_path, 'dataset=disk,extra_dims=0,loss=ce,loss_milestones=[0.5, 0.4, 0.3],max_its=25000,min_its=5000,n_runs=100.pkl'),
     lambda: generate_disk_dataset(n_train=1, p=0,
                                   test_resolution=test_res,
                                   device=device))
]

# %%

def create_toy_plot(path, ds_generate):
    fig = create_figure(0.33, 1.8)

    subfigs = fig.subfigures(1, 2, wspace=0, width_ratios=[1, 1.25])

    fig = subfigs[0]; axis = fig.gca()
    x_train, y_train, x_test, y_test = ds_generate()
    fig.suptitle('dataset')
    img = custom_imshow(axis, y_test)
    axis.set_xticks([]); axis.set_yticks([])

    from viz_utils import cmap

    plt.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(),
                marker='x', cmap=cmap, s=.1)

    fig = subfigs[1]; axis = fig.subplots(1, 1)
    fig.suptitle('$\Delta$loss')
    with open(path, 'rb') as f:
        stats_adaptv, r_adaptv, stats_linear, r_linear = pkl.load(f)

    test_loss_adaptv = filter(stats_adaptv, 'test_loss_indiv', 1)
    test_loss_linear = filter(stats_linear, 'test_loss_indiv', 1)
    mean_loss_adaptv = sum(test_loss_adaptv) / len(test_loss_adaptv)
    mean_loss_linear = sum(test_loss_linear) / len(test_loss_linear)

    img = custom_imshow(axis, mean_loss_adaptv - mean_loss_linear,
                        center=True)
    axis.set_xticks([]); axis.set_yticks([])
    plt.colorbar(img, ax=axis, orientation='vertical', shrink=.68)

    fname = '.'.join(path.split('/')[-1].split('.')[:-1])
    save_fig(plt.gcf(), os.path.join(fig_path, f'{fname}.pdf'))

for path, name in to_plot:
    create_toy_plot(path, name)
# %%

linewidth = .7
color_lin = 'tomato'
color_adaptv = 'lightgreen'

def create_train_plot(path, ds_generate, acc=False):
    create_figure(0.5, 2)

    if acc:
        key = 'train_acc'
        suffix = 'accuracy'
    else:
        key = 'train_loss'
        suffix = 'loss'
    with open(path, 'rb') as f:
        stats_adaptv, r_adaptv, stats_linear, r_linear = pkl.load(f)

    for r in r_adaptv:
        plt.plot(r.get(key), color=color_adaptv, alpha=.25, linewidth=linewidth)
    for r in r_linear:
        plt.plot(r.get(key), color=color_lin, alpha=.25, linewidth=linewidth)
    if acc:
        plt.ylim(.5, 1)
    else:
        plt.ylim(0, np.log(2))

    plt.xlim(50, 5000)
    xlims = plt.xlim()
    plt.plot([xlims[0], xlims[1]], [.4, .4], linewidth=linewidth, color='black')

    plt.ylabel(f'training {suffix}')
    plt.xlabel('GD iterations')
    plt.xscale('log')

    plt.plot([], [], linewidth=linewidth, color=color_lin,
             label='linearized ($\\alpha=100$)')
    plt.plot([], [], linewidth=linewidth, color=color_adaptv,
             label='non-linear ($\\alpha=1$)')
    plt.legend(loc='lower left')

    fname = '.'.join(path.split('/')[-1].split('.')[:-1])
    save_fig(plt.gcf(), os.path.join(fig_path, f'{fname}_{suffix}.pdf'))

for path, name in to_plot:
    create_train_plot(path, name, acc=True)
    create_train_plot(path, name, acc=False)

# %%
loss_milestones = [.5, .4, .3]
labelpad = 13

def create_full_plot(path, ds_generate):
    fig = create_figure(1, 6)
    spec = fig.add_gridspec(ncols=5, nrows=1, width_ratios=[1, 1.5, 1, 1, 1])
    
    ###
    axis = fig.add_subplot(spec[0, 1])
    with open(path, 'rb') as f:
        stats_adaptv, r_adaptv, stats_linear, r_linear = pkl.load(f)

    for r in r_adaptv:
        plt.plot(r.get('train_loss'), color=color_adaptv, alpha=.25, linewidth=linewidth)
    for r in r_linear:
        plt.plot(r.get('train_loss'), color=color_lin, alpha=.25, linewidth=linewidth)
    plt.ylim(0, np.log(2))

    plt.xlim(50, 5000)
    xlims = plt.xlim()
    for i, loss_milestone in enumerate(loss_milestones):
        plt.plot([xlims[0], xlims[1]], [loss_milestone, loss_milestone],
                 linewidth=linewidth, color='black', alpha=.8)
        plt.text(400+80*i, loss_milestone+.028, f'({"cde"[i]})',
                 size=5, ha="center", va="center")
    plt.ylabel('training loss')
    plt.xlabel('GD iterations')
    plt.xscale('log')

    plt.plot([], [], linewidth=linewidth, color=color_lin,
             label='linearized ($\\alpha=100$)')
    plt.plot([], [], linewidth=linewidth, color=color_adaptv,
             label='non-linear ($\\alpha=1$)')
    plt.legend(loc='lower left')

    ###
    axis = fig.add_subplot(spec[0, 0])
    x_train, y_train, x_test, y_test = ds_generate()
    img = custom_imshow(axis, y_test)
    axis.set_xticks([]); axis.set_yticks([])

    from viz_utils import cmap

    plt.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(),
                marker='x', cmap=cmap, s=.1)
    plt.xlabel('(a) task +\nexample dataset', labelpad=labelpad)

    ###
    with open(path, 'rb') as f:
        stats_adaptv, r_adaptv, stats_linear, r_linear = pkl.load(f)

    vmax = 0
    for i, loss_milestone in enumerate(loss_milestones):
        test_loss_adaptv = filter(stats_adaptv, 'test_loss_indiv', i+1)
        test_loss_linear = filter(stats_linear, 'test_loss_indiv', i+1)
        if len(test_loss_linear) == 0:
            continue
        mean_loss_adaptv = sum(test_loss_adaptv) / len(test_loss_adaptv)
        mean_loss_linear = sum(test_loss_linear) / len(test_loss_linear)
        vmax = max(vmax, torch.abs(mean_loss_adaptv - mean_loss_linear).max().item())

    for i, loss_milestone in enumerate(loss_milestones):
        axis = fig.add_subplot(spec[0, 2+i])
        test_loss_adaptv = filter(stats_adaptv, 'test_loss_indiv', i+1)
        test_loss_linear = filter(stats_linear, 'test_loss_indiv', i+1)
        if len(test_loss_linear) == 0:
            continue
        mean_loss_adaptv = sum(test_loss_adaptv) / len(test_loss_adaptv)
        mean_loss_linear = sum(test_loss_linear) / len(test_loss_linear)

        img = custom_imshow(axis, mean_loss_adaptv - mean_loss_linear,
                            center=True, vlims=vmax)
        axis.set_xticks([]); axis.set_yticks([])
        plt.xlabel(f'({"cde"[i]}) '
                   + '$\Delta loss\left(x_{test}\\right)$ at\ntraining loss='
                   + str(loss_milestone),
                   labelpad=labelpad)
        plt.colorbar(img, ax=axis, orientation='vertical', shrink=.85)

    ###
    fname = '.'.join(path.split('/')[-1].split('.')[:-1])
    save_fig(plt.gcf(), os.path.join(fig_path, f'{fname}_full.pdf'))

for path, name in to_plot:
    create_full_plot(path, name)

# %%
