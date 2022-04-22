# %%

import pickle as pkl
import sys
sys.path.append('..')
from plot_utils import *
import os
import numpy as np
from train_utils import Recorder

# %%

fig_path = '/network/projects/g/georgeth/linvsnonlin/celeba_figures/'

save_dir = '/network/projects/g/georgeth/linvsnonlin/celeba'
f_name = 'recorder_3'
pkl_path = os.path.join(save_dir, f'{f_name}.pkl')


# %%
alphas = [0.5, 1, 10]
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
    x, y = smoothen_lowess(x, y)
    # x, y = rotate_back(x, y, np.pi/4, (.5, .5))
    return x, y


# %%
for ds_name in ['trainb', 'test']:
    f = create_figure(3, 1)

    for r, alpha in zip(recorders, alphas):
        x, y = r.get(f'acc_unflipped_{ds_name}'), r.get(f'acc_flipped_{ds_name}')

        plt.scatter(x, y, alpha=.03, marker='+')

        x_s, y_s = smoothen_xy(x, y)
        plt.plot(x_s, y_s, label=f'alpha={alpha}')

    plt.ylabel(f'{ds_name} acc. balanced man')
    plt.xlabel(f'{ds_name} acc. balanced woman')
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

    plt.scatter(x, y, alpha=.03, marker='+')

    x_s, y_s = smoothen_xy(x, y)
    plt.plot(x_s, y_s, label=f'alpha={alpha}')
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
