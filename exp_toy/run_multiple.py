# %%
from ntk_alignment_viz_on_toy_datasets import main
from collections import namedtuple
from viz_utils import custom_imshow
import matplotlib.pyplot as plt
import time

Args = namedtuple('Args', 'n_train lr max_iterations l2 loss flipped test_resolution dataset alpha seed')
stats_adaptv = []
stats_linear = []

# %%
n_runs = 100
start_time = time.time()
for i in range(n_runs):
    args = Args(dataset='disk', # ['disk', 'disk_flip_vertical', 'yinyang1', 'yinyang2']
                n_train=50,
                lr=1e-1,
                max_iterations=3000,
                l2=0.,
                loss='ce',
                flipped=0.,
                test_resolution=60,
                alpha=1,
                seed=i)
    stats_adaptv.append(main(args, generate_plots=False, compute_jacobians=False))
    
    args = Args(dataset='disk', # ['disk', 'disk_flip_vertical', 'yinyang1', 'yinyang2']
                n_train=50,
                lr=1e-1,
                max_iterations=3000,
                l2=0.,
                loss='ce',
                flipped=0.,
                test_resolution=60,
                alpha=100,
                seed=i)
    stats_linear.append(main(args, generate_plots=True, compute_jacobians=False))

    print(f'run {i}, time={time.time() - start_time:.2f}s')

# %%

len(stats_adaptv), len(stats_linear)

# %%
mean_acc_adaptv = sum([s['test_acc_indiv'][2] for s in stats_adaptv])
custom_imshow(plt.gca(), mean_acc_adaptv)

# %%

mean_acc_linear = sum([s['test_acc_indiv'][2] for s in stats_linear])
custom_imshow(plt.gca(), mean_acc_linear)

# %%
custom_imshow(plt.gca(), mean_acc_adaptv - mean_acc_linear)

# %%
mean_loss_adaptv = sum([s['test_loss_indiv'][2] for s in stats_adaptv])
custom_imshow(plt.gca(), mean_loss_adaptv)

# %%

mean_loss_linear = sum([s['test_loss_indiv'][2] for s in stats_linear])
custom_imshow(plt.gca(), mean_loss_linear)

# %%

img = custom_imshow(plt.gca(), mean_loss_adaptv - mean_loss_linear)
plt.colorbar(img)
plt.xticks([]); plt.yticks([])

# %%

for k, v in stats_adaptv[0].items():
    try:
        print(k, v[0].size())
    except:
        pass
# %%