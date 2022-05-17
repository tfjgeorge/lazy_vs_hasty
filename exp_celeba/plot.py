# %%

import pickle as pkl
import sys
sys.path.append('..')
from plot_utils import *
import os
import numpy as np
from train_utils import Recorder
from plot_utils import smoothen_running_average

# %%

fig_path = '/network/projects/g/georgeth/linvsnonlin/celeba_figures/'

save_dir = '/network/projects/g/georgeth/linvsnonlin/celeba'
f_name = 'recorder_longer5'
pkl_path = os.path.join(save_dir, f'{f_name}.pkl')


# %%
alphas = [0.5, 1, 100]
recorders = pkl.load(open(pkl_path, 'rb'))

# %%


import statsmodels.api as sm
def smoothen_lowess(x, y):
    lowess = sm.nonparametric.lowess(y, x, frac=.15)
    x = lowess[:, 0]
    y = lowess[:, 1]
    return x, y

def rotate(x, y, angle=np.pi/4, origin=(.5, .5)):
    x = x - origin[0]
    y = y - origin[1]
    x_prime = x * np.cos(angle) + y * np.sin(angle)
    y_prime = -x * np.sin(angle) + y * np.cos(angle)
    return x_prime, y_prime

def rotate_back(x, y, angle=np.pi/4, origin=(.5, .5)):
    x_prime = x * np.cos(-angle) + y * np.sin(-angle)
    y_prime = -x * np.sin(-angle) + y * np.cos(-angle)
    return x_prime + origin[0], y_prime + origin[1]

def smoothen_xy(x, y):
    x = np.array(x)
    y = np.array(y)
    # x, y = rotate(x, y, np.pi/4, (.5, .5))
    x = smoothen_running_average(.9)(x)
    y = smoothen_running_average(.9)(y)
    # x, y = rotate_back(x, y, np.pi/4, (.5, .5))
    return x, y


def plot(axis, x, y, label):
    # axis.scatter(x, y, alpha=1, marker='.')

    x_s, y_s = smoothen_xy(x, y)
    axis.plot(x_s, y_s, label=f'alpha={alpha}', alpha=.8)

# %%
for ds_name, name_readable in zip(['trainb', 'test'], ['train', 'test']):
    f = create_figure(3, 1)

    for r, alpha in zip(recorders, alphas):
        x, y = r.get(f'acc_unflipped_{ds_name}'), r.get(f'acc_flipped_{ds_name}')
        x = np.insert(x, 0, .5)
        y = np.insert(y, 0, .5)
        plot(plt.gca(), x, y, f'alpha={alpha}')

    plt.ylabel(f'{name_readable} acc. balanced man')
    plt.xlabel(f'{name_readable} acc. balanced woman')
    plt.legend()
    plt.ylim(.5, 1)
    plt.xlim(.5, 1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    save_fig(f, os.path.join(fig_path, f'{f_name}_{ds_name}_acc_spur_vs_actual.pdf'))
    plt.show()

# %%
f = create_figure(3, 1)

for r, alpha in zip(recorders, alphas):
    x, y = r.get(f'acc'), r.get(f'acc_test')
    x = np.insert(x, 0, .5)
    y = np.insert(y, 0, .5)
    plot(plt.gca(), x, y, f'alpha={alpha}')
    
plt.xlabel('acc train')
plt.ylabel('acc test')
plt.legend()
# plt.ylim(1e-1, 3)
plt.xlim(.75, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
save_fig(f, os.path.join(fig_path, f'{f_name}_acc_test_vs_train.pdf'))
plt.show()

# %%

f = create_figure(3, 1)
colors = dict()

for r, alpha in zip(recorders, alphas):
    x, y = r.get(f'loss_100'), r.get(f'sign_similarity')
    x = np.insert(x, 0, np.log(2))
    y = np.insert(y, 0, 1)
    x_s, y_s = smoothen_xy(x, y)
    p = plt.gca().plot(x_s, y_s, marker='x')
    c = p[0].get_color()
    colors[alpha] = c

    x, y = r.get(f'loss_100'), r.get(f'ntk_alignment')
    x = np.insert(x, 0, np.log(2))
    y = np.insert(y, 0, 1)
    x_s, y_s = smoothen_xy(x, y)
    plt.gca().plot(x_s, y_s, marker='^', color=c)

    x, y = r.get(f'loss_100'), r.get(f'repr_alignment')
    x = np.insert(x, 0, np.log(2))
    y = np.insert(y, 0, 1)
    x_s, y_s = smoothen_xy(x, y)
    plt.gca().plot(x_s, y_s, marker='d', color=c)

for alpha, c in colors.items():
    plt.plot([], [], color=c, label=f'alpha={alpha}')
plt.plot([], [], color='black', marker='x', label=f'sign similarity')
plt.plot([], [], color='black', marker='^', label=f'ntk alignment')
plt.plot([], [], color='black', marker='d', label=f'repr. kernel align.')

plt.xlabel('loss train')
plt.ylabel('linearity measures')
plt.legend()
# plt.ylim(1e-1, 3)
xlims = plt.xlim()
plt.xlim(xlims[1], xlims[0])
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
save_fig(f, os.path.join(fig_path, f'{f_name}_lin_vs_loss.pdf'))
plt.show()

# %%
