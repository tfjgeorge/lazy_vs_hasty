# %%

import numpy as np
import matplotlib.pyplot as plt

import sys

from sklearn.cluster import k_means
sys.path.append('..')
from plot_utils import *

# %%

####################################################################
## Example 1

t_lin_max = 250
t_nonlin_max = 1
res = 250
sigma = 0.001

def generate_example1():
    mus = np.array([9, 8, 7, 6, 5])[np.newaxis, :]
    xs = np.diagflat(mus**.5)

    ys = mus[::-1] / mus**.5
    return xs, ys, mus

xs, ys, mus = generate_example1()
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

####################################################################
## Example 2

n = 20 # number of training examples
q = 5 # amongst them, number of randomly flipped examples
eta = 5 # noise magnitude
sigma = 1e-3 # initialization scale

d = 200 # number of dimensions

def generate_example2(n, q, eta, d):
    kappas = np.ones(n)
    shuffled_indices = np.arange(n)
    np.random.shuffle(shuffled_indices)
    kappas[shuffled_indices[:q]] = -1

    ys = np.random.randint(0, 2, size=n) * 2 - 1
    ys_noisy = kappas * ys
    xs = np.concatenate([ys[:, np.newaxis],
                        np.diag([eta]*n),
                        np.zeros((n, d - n - 1))], axis=1)
    return xs, ys, ys_noisy, kappas

xs, ys, ys_noisy, kappas = generate_example2(n, q, eta, d)
# xs = np.concatenate([ys[:, np.newaxis],
#                      np.diag([eta]*n)], axis=1)
                    
u, sqrt_mus, vT = np.linalg.svd(xs, full_matrices=True)
mus = np.concatenate([sqrt_mus**2, np.zeros(vT.shape[0] - sqrt_mus.shape[0])])

t_lin_max = 50
t_nonlin_max = 1
res = 250 # resolution of time scale

t_lin = np.linspace(0, t_lin_max, res)[:, np.newaxis]
t_nonlin = np.linspace(0, t_nonlin_max, res)[:, np.newaxis]

theta_0 = np.ones(xs.shape[1]) * sigma
theta_star, _, _, _ = np.linalg.lstsq(xs, ys_noisy)
theta_alpha_star = np.dot(vT, theta_star)
vT = np.sign(theta_alpha_star)[:, None] * vT
theta_alpha_star = np.abs(theta_alpha_star)
# theta_alpha_0 = np.dot(vT, theta_0)
theta_alpha_0 = np.ones(theta_alpha_star.shape[0]) * sigma

ys_tilda = np.dot(vT, np.dot(xs.T, ys_noisy))
ys_tilda = np.abs(ys_tilda)

theta_alpha_lin = theta_alpha_star + np.exp(-2 * mus * theta_alpha_0 * t_lin) * (theta_alpha_0 - theta_alpha_star)
theta_alpha_nonlin = theta_alpha_star + theta_alpha_star * np.nan_to_num((theta_alpha_0 - theta_alpha_star) / (np.exp(2 * ys_tilda * t_nonlin) * theta_alpha_0 -(theta_alpha_0 - theta_alpha_star)))

# project back:
theta_lin = np.dot(theta_alpha_lin, vT)
theta_nonlin = np.dot(theta_alpha_nonlin, vT)

err_lin = np.dot(theta_lin, xs.T) - ys_noisy
err_nonlin = np.dot(theta_nonlin, xs.T) - ys_noisy

err_lin = err_lin**2
err_nonlin = err_nonlin**2

err_lin_clean = err_lin[:, kappas==1].sum(axis=1) / (n - q)
err_lin_noisy = err_lin[:, kappas==-1].sum(axis=1) / q
err_nonlin_clean = err_nonlin[:, kappas==1].sum(axis=1) / (n - q)
err_nonlin_noisy = err_nonlin[:, kappas==-1].sum(axis=1) / q

#

fig = create_figure(.33, 1.5)

x_lin = np.arange(len(err_lin_clean))
x_nonlin = np.arange(len(err_nonlin_clean))
# x_lin = q/n * err_lin_noisy + (n-q)/n * err_lin_clean
# x_nonlin = q/n * err_nonlin_noisy + (n-q)/n * err_nonlin_clean
# x_lin = err_lin_clean
# x_nonlin = err_nonlin_clean

p = plt.plot(x_lin, err_lin_clean)
color_lin = p[0].get_color()
p = plt.plot(x_nonlin, err_nonlin_clean)
color_nonlin = p[0].get_color()

plt.plot(x_lin, err_lin_noisy, '--', c=color_lin)
plt.plot(x_nonlin, err_nonlin_noisy, '--', c=color_nonlin)

plt.plot([], [], color='black', label='clean')
plt.plot([], [], '--', color='black', label='noisy')
plt.plot([], [], '-', color=color_lin, label='linear')
plt.plot([], [], '-', color=color_nonlin, label='non-linear')
plt.legend(ncol=1)
plt.ylabel('Clean/noisy examples MSE')
plt.xlabel('time (arbitrary scale)')
# plt.yscale('log')
plt.ylim(0, 2.)

# xlims = plt.xlim()
# plt.xlim(xlims[1], xlims[0])

save_fig(fig, f'example2.pdf')

# %%

####################################################################
## Example 3

n = 20 # number of training examples
q = 5 # amongst them, examples that do not have the spurious correlation
eta = 10 # noise magnitude
sigma = 1e-3 # initialization scale
lambd = 1 # spurious correlation strength

d = 200 # number of dimensions

t_lin_max = 10
t_nonlin_max = .2
res = 250 # resolution of time scale

t_lin = np.linspace(0, t_lin_max, res)[:, np.newaxis]
t_nonlin = np.linspace(0, t_nonlin_max, res)[:, np.newaxis]

kappas = np.ones(n) # kappa = 1: spur correl, kappa = -1 no spur correl
shuffled_indices = np.arange(n)
np.random.shuffle(shuffled_indices)
kappas[shuffled_indices[:q]] = -1

ys = np.random.randint(0, 2, size=n) * 2 - 1
ys_noisy = kappas * ys
xs = np.concatenate([ys[:, np.newaxis],
                     (lambd * kappas * ys)[:, np.newaxis],
                     np.diag([eta]*n),
                     np.zeros((n, d - n - 2))], axis=1)
# xs = np.concatenate([ys[:, np.newaxis],
#                      np.diag([eta]*n)], axis=1)
                    
u, sqrt_mus, vT = np.linalg.svd(xs, full_matrices=True)
mus = np.concatenate([sqrt_mus**2, np.zeros(vT.shape[0] - sqrt_mus.shape[0])])

theta_0 = np.ones(xs.shape[1]) * sigma
theta_star, _, _, _ = np.linalg.lstsq(xs, ys_noisy)
theta_alpha_star = np.dot(vT, theta_star)
vT = np.sign(theta_alpha_star)[:, None] * vT
theta_alpha_star = np.abs(theta_alpha_star)
# theta_alpha_0 = np.dot(vT, theta_0)
theta_alpha_0 = np.ones(theta_alpha_star.shape[0]) * sigma

ys_tilda = np.dot(vT, np.dot(xs.T, ys_noisy))
ys_tilda = np.abs(ys_tilda)

theta_alpha_lin = theta_alpha_star + np.exp(-2 * mus * theta_alpha_0 * t_lin) * (theta_alpha_0 - theta_alpha_star)
theta_alpha_nonlin = theta_alpha_star + theta_alpha_star * np.nan_to_num((theta_alpha_0 - theta_alpha_star) / (np.exp(2 * ys_tilda * t_nonlin) * theta_alpha_0 -(theta_alpha_0 - theta_alpha_star)))

# project back:
theta_lin = np.dot(theta_alpha_lin, vT)
theta_nonlin = np.dot(theta_alpha_nonlin, vT)

err_lin = np.dot(theta_lin, xs.T) - ys_noisy
err_nonlin = np.dot(theta_nonlin, xs.T) - ys_noisy

err_lin = err_lin**2
err_nonlin = err_nonlin**2

err_lin_spur = err_lin[:, kappas==1].sum(axis=1) / (n - q)
err_lin_nospur = err_lin[:, kappas==-1].sum(axis=1) / q
err_nonlin_spur = err_nonlin[:, kappas==1].sum(axis=1) / (n - q)
err_nonlin_nospur = err_nonlin[:, kappas==-1].sum(axis=1) / q

#

fig = create_figure(.33, 1.5)

x_lin = np.arange(len(err_lin_spur))
x_nonlin = np.arange(len(err_nonlin_spur))
# x_lin = q/n * err_lin_nospur + (n-q)/n * err_lin_spur
# x_nonlin = q/n * err_nonlin_nospur + (n-q)/n * err_nonlin_spur
# x_lin = err_lin_spur
# x_nonlin = err_nonlin_spur

p = plt.plot(x_lin, err_lin_spur)
color_lin = p[0].get_color()
p = plt.plot(x_nonlin, err_nonlin_spur)
color_nonlin = p[0].get_color()

plt.plot(x_lin, err_lin_nospur, '--', c=color_lin)
plt.plot(x_nonlin, err_nonlin_nospur, '--', c=color_nonlin)

plt.plot([], [], color='black', label='spur.')
plt.plot([], [], '--', color='black', label='no spur.')
plt.plot([], [], '-', color=color_lin, label='linear')
plt.plot([], [], '-', color=color_nonlin, label='non-linear')
plt.legend(ncol=1)
plt.ylabel('Per group MSE')
plt.xlabel('time (arbitrary scale)')
# plt.yscale('log')
plt.ylim(0, 2.)

# xlims = plt.xlim()
# plt.xlim(xlims[1], xlims[0])

save_fig(fig, f'example3.pdf')
# %%
