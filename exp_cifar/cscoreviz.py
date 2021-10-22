# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch


# %%
with open('cifar10-cscores-orig-order.npz', 'rb') as f:
    d = np.load(f)
    cscores = d['scores']
    labels = d['labels']


# %%
def margin(logits, targets):
    logits_target = logits[np.arange(len(logits)), targets]
    logits_cp = logits.clone()

    logits_cp[np.arange(len(logits)), targets] = -1e10
    margins = (logits_target.unsqueeze(1) - logits_cp).min(1)[0]

    return margins

# %%

def margin_mean(logits, targets):
    logits_target = logits[np.arange(len(logits)), targets]
    logits_cp = logits.clone()

    logits_cp[np.arange(len(logits)), targets] = 0
    margins = logits_target - logits_cp.sum(dim=1) / 9

    return margins


# %%
cscores = cscores[:1000]
labels = labels[:1000]

cscores_order = cscores.argsort()
cscores_ranks = cscores_order.argsort()


# %%
plt.hist(cscores, bins=25)


# %%
path = 'results/alpha=1.0,depth=0,diff=0.0,diff_type=random,epochs=30,l2=0.0,lr=0.02,mom=0.0,seed=1,task=cifar10_resnet18,width=0/'


# %%
d = pd.read_pickle(os.path.join(path, 'lin_log.pkl'))

d.shape


# %%
d_log = pd.read_pickle(os.path.join(path, 'log.pkl'))


# %%
n_plot = d_log.shape[0] // 4
n_plot


# %%
plt.plot(d_log['iteration'], d_log['train_acc'])
plt.plot(d_log['iteration'], d_log['test_acc'])


# %%
score2s = []
logits = []

for i, l in d.iterrows():

    logits.append(l['train_nonlin'] - l['train_lin'].t())
    score2 = torch.norm(logits[-1], dim=1)
    score2s.append(score2)

    if i % n_plot == 0:
        plt.figure()
        plt.scatter(1 - cscores, score2, alpha=.5)
        plt.xscale('log')
        plt.show()


# %%
for i, l in d.iterrows():

    logits_this = l['train_nonlin'] - l['train_lin'].t()

    if i % n_plot == 0:
        plt.figure()
        plt.scatter(1 - cscores, logits_this[np.arange(1000), labels], alpha=.5)
        plt.xscale('log')
        plt.show()


# %%
for i, l in d.iterrows():

    logits_this = l['train_nonlin'] - l['train_lin'].t()

    if i % n_plot == 0:
        plt.figure()
        plt.scatter(cscores_ranks, logits_this[np.arange(1000), labels], alpha=.5)
        # plt.xscale('log')
        plt.show()


# %%
for i, l in d.iterrows():

    logits_this = l['train_nonlin'] - l['train_lin'].t()

    if i % n_plot == 0:
        plt.figure()
        plt.scatter(cscores_ranks, margin(logits_this, labels), alpha=.5)
        # plt.xscale('log')
        plt.title(l['iteration'])
        plt.grid()
        plt.show()


# %%
plt.scatter(1 - cscores, sum(score2s), alpha=.5)
plt.xscale('log')
plt.show()


# %%
sum_logits = sum(logits)
all_sum_logits = []

for i in range(4):
    all_sum_logits.append(sum(logits[:min((i+1)*n_plot, len(logits))]))


# %%
plt.scatter(1 - cscores, torch.norm(sum_logits, dim=1), alpha=.5)
# plt.xscale('log')
plt.show()


# %%
plt.scatter(1 - cscores, sum_logits[np.arange(1000), labels], alpha=.5)
# plt.xscale('log')
plt.show()


# %%
plt.scatter(cscores_ranks, sum_logits[np.arange(1000), labels], alpha=.5)
# plt.xscale('log')
plt.show()


# %%
plt.scatter(cscores_ranks, sum_logits[np.arange(1000), labels] - sum_logits[np.arange(1000)].max(axis=1)[0], alpha=.5)
plt.xlabel('c scores')
plt.ylabel('margin(f_nonlin - f_lin)')
# plt.xscale('log')
plt.show()


# %%
plt.scatter(cscores_ranks, margin(sum_logits, labels), alpha=.5)
plt.xlabel('c score rank')
plt.ylabel('margin(f_nonlin - f_lin)[y]')
# plt.xscale('log')
plt.show()

# %%
plt.scatter(cscores_ranks, margin_mean(sum_logits, labels), alpha=.5)
plt.xlabel('c score rank')
plt.ylabel('margin mean(f_nonlin - f_lin)[y]')
# plt.xscale('log')
plt.show()

# %%

for i in range(4):
    plt.figure()
    it = int(d_log['iteration'].iloc[(i+1)*n_plot])
    plt.title(f'{it} iterations')
    plt.scatter(cscores_ranks, margin_mean(all_sum_logits[i], labels), alpha=.3)
    plt.xlabel('c score rank')
    plt.ylabel('margin mean(f_nonlin - f_lin)[y]')
    plt.grid()
    plt.show()


# %%
plt.scatter(cscores_ranks, margin(-sum_logits, labels), alpha=.5)
plt.xlabel('c score rank')
plt.ylabel('margin(f_lin - f_nonlin)')
# plt.xscale('log')
plt.show()