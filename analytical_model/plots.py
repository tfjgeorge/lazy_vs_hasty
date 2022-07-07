# %%

import numpy as np
import matplotlib.pyplot as plt

import sys

from sklearn.cluster import k_means
sys.path.append('..')
from plot_utils import *

# %%

## Example 1

t_lin_max = 250
t_nonlin_max = 1
res = 250

sigma = 0.001

mus = np.array([9, 8, 7, 6, 5])[np.newaxis, :]

ys = mus[::-1] / mus**.5
ys_tilda = mus**.5 * ys

t_lin = np.linspace(0, t_lin_max, res)[:, np.newaxis]
t_nonlin = np.linspace(0, t_nonlin_max, res)[:, np.newaxis]

theta_0 = np.ones(mus.shape[0]) * sigma
theta_star = np.ones(mus.shape[0])

theta_lin = theta_star + np.exp(-2 * mus[:, ::-1] * theta_0 * t_lin) * (theta_0 - theta_star)
theta_nonlin = theta_star + np.exp(-2 * ys_tilda * t_nonlin) * (theta_star * (theta_0 - theta_star)) / (theta_0 - np.exp(-2 * ys_tilda * t_nonlin) * (theta_0 - theta_star))

err_lin = mus**.5 * theta_lin - ys
err_nonlin = mus**.5 * theta_nonlin - ys

err_lin = err_lin**2
err_nonlin = err_nonlin**2
# %%

fig = create_figure(.33, 1.5)
p = plt.plot(err_lin)
for i in range(err_nonlin.shape[1]):
    plt.plot(err_nonlin[:, i], '--', c=p[i].get_color())

plt.plot([], [], color='black', label='linear')
plt.plot([], [], '--', color='black', label='non-linear')
plt.legend()
plt.ylabel('per example squared error')
plt.xlabel('time (arbitrary scale)')
# plt.yscale('log')

save_fig(fig, f'example1.pdf')

# %%

## Example 2

n = 20 # number of training examples
q = 5 # amongst them, number of randomly flipped examples
eta = .1 # noise magnitude
sigma = 0.001 # initialization scale
res = 250 # resolution of time scale

d = 25 # number of dimensions

kappas = np.ones(n)
shuffled_indices = np.arange(n)
np.random.shuffle(shuffled_indices)
kappas[shuffled_indices[:q]] = -1

ys = np.random.randint(0, 2, size=n) * 2 - 1
xs = np.concatenate([(kappas * ys)[:, np.newaxis],
                     np.diag([eta]*n),
                     np.zeros((n, d - n - 1))], axis=1)
                    
u, sqrt_mus, vT = np.linalg.svd(xs, full_matrices=False)
mus = sqrt_mus**2

t_lin_max = 250
t_nonlin_max = 250

t_lin = np.linspace(0, t_lin_max, res)[:, np.newaxis]
t_nonlin = np.linspace(0, t_nonlin_max, res)[:, np.newaxis]

theta_0 = np.ones(mus.shape[0]) * sigma
theta_star, _, _, _ = np.linalg.lstsq(xs, ys)
theta_alpha_star = np.dot(vT, theta_star)

ys_tilda = np.dot(vT, np.dot(xs.T, ys))

theta_alpha_lin = theta_alpha_star + np.exp(-2 * mus * theta_0 * t_lin) * (theta_0 - theta_alpha_star)
theta_alpha_nonlin = theta_alpha_star + np.exp(-2 * ys_tilda * t_nonlin) * (theta_alpha_star * (theta_0 - theta_alpha_star)) / (theta_0 - np.exp(-2 * ys_tilda * t_nonlin) * (theta_0 - theta_alpha_star))

# project back:
theta_lin = np.dot(theta_alpha_lin, vT)
theta_nonlin = np.dot(theta_alpha_nonlin, vT)

err_lin = np.dot(theta_lin, xs.T) - ys
err_nonlin = np.dot(theta_nonlin, xs.T) - ys

err_lin = err_lin**2
err_nonlin = err_nonlin**2

err_lin_clean = err_lin[:, kappas==1].sum(axis=1) / (n - q)
err_lin_noisy = err_lin[:, kappas==-1].sum(axis=1) / q
err_nonlin_clean = err_nonlin[:, kappas==1].sum(axis=1) / (n - q)
err_nonlin_noisy = err_nonlin[:, kappas==-1].sum(axis=1) / q

#

fig = create_figure(.33, 1.5)

p = plt.plot(err_lin_clean)
color_clean = p[0].get_color()
p = plt.plot(err_lin_noisy)
color_noisy = p[0].get_color()

plt.plot(err_nonlin_clean, '--', c=color_clean)
plt.plot(err_nonlin_noisy, '--', c=color_noisy)

plt.plot([], [], color='black', label='linear')
plt.plot([], [], '--', color='black', label='non-linear')
plt.legend()
plt.ylabel('per example squared error')
plt.xlabel('time (arbitrary scale)')
# plt.yscale('log')

save_fig(fig, f'example2.pdf')


# %%
