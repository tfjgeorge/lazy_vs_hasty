import numpy as np
import pickle as pkl
from viz_utils import custom_imshow
import matplotlib.pyplot as plt

dataset = 'disk' # ['disk', 'disk_flip_vertical', 'disk_flip_diagonal', 'yinyang1', 'yinyang2']
n_runs = 100
loss = 'ce'
extra_dims = 0
max_iterations = 5000

fname = f'{dataset}_{extra_dims}_{loss}_{n_runs}_{max_iterations}'

# %%

with open(f'/network/projects/g/georgeth/linvsnonlin/toy/{fname}.pkl', 'rb') as f:
    stats_adaptv, stats_linear = pkl.load(f)


# %%

print(np.unique([len(s['test_acc_indiv']) for s in stats_linear], return_counts=True))
print(np.unique([len(s['test_acc_indiv']) for s in stats_adaptv]))
min_i = np.min([len(s['test_acc_indiv']) for s in stats_linear])
# %%
i = min_i-1

# %%
mean_acc_adaptv = sum([s['test_acc_indiv'][i] for s in stats_adaptv])
custom_imshow(plt.gca(), mean_acc_adaptv)

# %%

mean_acc_linear = sum([s['test_acc_indiv'][i] for s in stats_linear])
custom_imshow(plt.gca(), mean_acc_linear)

# %%
custom_imshow(plt.gca(), mean_acc_adaptv - mean_acc_linear)

# %%
mean_loss_adaptv = sum([s['test_loss_indiv'][i] for s in stats_adaptv])
custom_imshow(plt.gca(), mean_loss_adaptv)

# %%

mean_loss_linear = sum([s['test_loss_indiv'][i] for s in stats_linear])
custom_imshow(plt.gca(), mean_loss_linear)

# %%

img = custom_imshow(plt.gca(), mean_loss_adaptv - mean_loss_linear)
plt.colorbar(img)
plt.xticks([]); plt.yticks([])

# %%

def filter(a, f, i):
    o = []
    for x in a:
        if len(x[f]) > i+1:
            o.append(x[f][i])
    return o

f = plt.figure(figsize=(min_i*4.5, 2*3.5))
f.patch.set_facecolor('xkcd:white')
plt.suptitle(f'loss difference (top row) and std (bottom row) adaptative - linear\nat various points during training (from left to right)')

for i in range(min_i):
    test_loss_adaptv = filter(stats_adaptv, 'test_loss_indiv', i)
    test_loss_linear = filter(stats_linear, 'test_loss_indiv', i)
    mean_loss_adaptv = sum(test_loss_adaptv) / len(test_loss_adaptv)
    mean_loss_linear = sum(test_loss_linear) / len(test_loss_linear)

    plt.subplot2grid((2, min_i), (0, i))
    img = custom_imshow(plt.gca(), mean_loss_adaptv - mean_loss_linear, center=True)
    plt.colorbar(img)
    plt.xticks([]); plt.yticks([])

    try:
        std_loss = np.std([(sa['test_loss_indiv'][i] - sl['test_loss_indiv'][i]).cpu().numpy()
                        for sa, sl in zip(stats_adaptv, stats_linear)], axis=0)

        plt.subplot2grid((2, min_i), (1, i))
        img = custom_imshow(plt.gca(), std_loss)
        plt.colorbar(img)
        plt.xticks([]); plt.yticks([])
    except:
        pass
 
plt.savefig(f'figures/{fname}.pdf')

# %%

np.std([s['test_loss_indiv'][i].cpu().numpy() for s in stats_adaptv], axis=0)
# %%

plt.figure(figsize=(4, 4))
cmap = plt.get_cmap('viridis')

for s_l, s_a in zip(stats_linear, stats_adaptv):

    if len(s_l['delta_loss_ex0']) <= 2:
        continue

    x_train = s_l['x_train'].cpu().detach()
    delta_loss = s_a['delta_loss_ex0'][1].cpu().detach()
    print(delta_loss.mean(), s_a['delta_loss_ex0'][1].cpu().detach().mean())
    plt.scatter(x_train[0, 0], x_train[0, 1], color=cmap(delta_loss.mean().item() / 5))

    print(s_l.keys())
    print(len(s_l['delta_loss_ex0']))

# %%
