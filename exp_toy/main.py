# %%

from collections import namedtuple

Args = namedtuple('Args', 'n_train lr max_iterations l2 loss flipped test_resolution dataset alpha seed')
args = Args(dataset='disk', # ['disk', 'disk_flip_vertical', 'yinyang1', 'yinyang2']
            n_train=50,
            lr=1e-1,
            max_iterations=50000,
            l2=0.,
            loss='ce',
            flipped=0.,
            test_resolution=60,
            alpha=1000,
            seed=1111)

import torch
import torch.nn as nn
import copy
import numpy as np
import argparse
from torch.utils.data import (TensorDataset, DataLoader)
from nngeometry.generator import Jacobian
from nngeometry.object import PushForwardDense, PVector
import pickle as pkl
import matplotlib.pyplot as plt

from datasets import generate_disk_dataset, generate_yinyang_dataset
from viz_utils import custom_imshow

acc_step = .125

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using device: {device}')

def gen_model():
    lin1 = nn.Linear(2, 60)
    lin2 = nn.Linear(60, 60)
    lin3 = nn.Linear(60, 60)
    lin4 = nn.Linear(60, 1)
    model = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3, nn.ReLU(), lin4)
    model.to(device)
    return model

class Recorder():
    def __init__(self):
        self.values = dict()

    def save(self, key, val, i=None):
        if i is not None:
            val = (i, val)
        if key in self.values.keys():
            self.values[key].append(val)
        else:
            self.values[key] = [val]

    def get(self, key):
        return self.values[key]

def get_jacobian(x, model):
    loader = DataLoader(TensorDataset(x), batch_size=x.size(0))
    pf = PushForwardDense(Jacobian(model), examples=loader)
    return pf.get_dense_tensor()[0]

iterations_record = np.round(10. ** np.arange(0, 10, .1))
def do_record(it):
    return it in iterations_record

def train(recorder, model, model_0, lr, alpha, iterations, x_train, y_train,
          x_test, y_test, loss_fn, l2_reg=0., compute_jacobians=False):
    
    stats = {
        'iterations': [],
        'train_acc': [],
        'test_acc_indiv': [],
        'test_loss_indiv': [],
        'f_hat_test': [],
        'jacobian': [],
    }
    next_acc_milestone = .5

    for i in range(iterations+1):

        w = PVector.from_model(model)

        with torch.no_grad():
            f_hat_test = model(x_test)[:, 0] - model_0(x_test)[:, 0]
            loss_test, _ = loss_fn(f_hat_test, y_test, alpha)
            acc_test_indivs = ((y_test * f_hat_test) > 0).float()
            recorder.save('test_loss', loss_test.cpu())
            recorder.save('test_acc', acc_test_indivs.mean().item())

        if i > 0 and (do_record(i) or i == iterations) :
            print(f'iteration {i}, train_loss: {loss_train.mean().item()}, train_acc: {acc_train.item()}')

        f_hat_train = model(x_train)[:, 0] - model_0(x_train)[:, 0]
        model.zero_grad()

        loss_train, grad_f = loss_fn(f_hat_train, y_train, alpha)
        acc_train = ((y_train * f_hat_train) > 0).float().mean()

        recorder.save('train_loss', loss_train.cpu())
        recorder.save('train_acc', acc_train.item())

        if i == 0 or recorder.get('train_acc')[-1] >= next_acc_milestone or i == (iterations - 1):
            # extract test ntk
            print(f'Extracting tangent features at iteration={i} for acc={next_acc_milestone}')
            if compute_jacobians:
                stats['jacobian'].append(get_jacobian(x_test, model))
            stats['iterations'].append(i)
            stats['train_acc'].append(recorder.get('train_acc')[-1])
            stats['test_acc_indiv'].append(acc_test_indivs)
            stats['test_loss_indiv'].append(loss_test)
            stats['f_hat_test'].append(f_hat_test)
            next_acc_milestone += acc_step

        # compute gradients
        f_hat_train.backward(grad_f)
        grad_w = PVector.from_model_grad(model).detach()

        # update:
        ((1 - lr * l2_reg) * w.detach() - lr/alpha * grad_w).copy_to_model(model)

    return stats

def mseloss(pred, target, alpha):
    loss = .5 * ((pred - target)**2).mean()
    grad_f = (pred - target) / pred.size(0)
    return loss, grad_f

def celoss(pred, target, alpha):
    target = (target + 1) / 2 # to 0-1
    pred = alpha * pred
    sigm = torch.sigmoid(pred)
    log_sigm = torch.nn.functional.logsigmoid(pred)
    log_1_minus_sigm = torch.nn.functional.logsigmoid(-pred)
    loss_examples = -(target * log_sigm + (1 - target) * log_1_minus_sigm)
    grad_f = (sigm - target) / pred.size(0)
    return loss_examples, grad_f

def main(args, generate_plots=False, compute_jacobians=False):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed+1)

    if args.dataset == 'yinyang1':
        x_train, y_train, x_test, y_test = generate_yinyang_dataset(n_train=args.n_train, p=args.flipped,
                                                                    test_resolution=args.test_resolution, variant=1,
                                                                    device=device)
    elif args.dataset == 'yinyang2':
        x_train, y_train, x_test, y_test = generate_yinyang_dataset(n_train=args.n_train, p=args.flipped,
                                                                    test_resolution=args.test_resolution, variant=2,
                                                                    device=device)
    elif args.dataset == 'disk':
        x_train, y_train, x_test, y_test = generate_disk_dataset(n_train=args.n_train, p=args.flipped,
                                                                test_resolution=args.test_resolution,
                                                                device=device)
    elif args.dataset == 'disk_flip_vertical':
        x_train, y_train, x_test, y_test = generate_disk_dataset(n_train=args.n_train, p=args.flipped,
                                                                test_resolution=args.test_resolution, half=True,
                                                                device=device)
    cmap = 'RdYlGn'
    
    if generate_plots:
        plt.figure()
        custom_imshow(plt.gca(), y_test)
        plt.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x',
                    cmap=cmap)
        plt.xticks([]); plt.yticks([])
        plt.show()


    model = gen_model()
    model_0 = copy.deepcopy(model)
    model.train()
    recorder = Recorder()

    if args.loss == 'mse':
        loss_fn = mseloss
    elif args.loss == 'ce':
        loss_fn = celoss

    stats = train(recorder, model, model_0, args.lr, args.alpha, args.max_iterations,
                  x_train, y_train, x_test, y_test, loss_fn, l2_reg=args.l2,
                  compute_jacobians=compute_jacobians)

    if generate_plots:
        plt.figure()
        plt.plot([l.mean().item() for l in recorder.get('train_loss')], label='train')
        plt.plot([l.mean().item() for l in recorder.get('test_loss')], label='test')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(recorder.get('train_acc'), label='train')
        plt.plot(recorder.get('test_acc'), label='test')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.grid()
        plt.show()

        plt.figure()
        custom_imshow(plt.gca(), 2. * (ntks['f_hat_test'][-1] > 0) - 1)
        plt.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(),
                    marker='x', cmap=cmap)
        plt.xticks([]); plt.yticks([])
        plt.show()

        usvs = []

        for i, j in enumerate(ntks['jacobian']):
            print(f'Computing eigendecomposition ({i+1}/{len(ntks["jacobian"])})')
            u, s, v = torch.svd(j.t() / j.size(0))
            usvs.append((u, s, v))

        mult = 3.5
        evals_i = [0, 1, 10, 100]

        fig = plt.figure(figsize=(17, 22), constrained_layout=True)
        subfigs_rows = fig.subfigures(2, 1, height_ratios=[1, 4])
        axes_row1 = subfigs_rows[0].subplots(1, 3)

        fig.suptitle(args.dataset)

        ## dataset
        axis = axes_row1[0]
        custom_imshow(axis, y_test)
        axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title('true distrib + dataset')

        ## training
        axis = axes_row1[1]

        axis.plot([l.mean().item() for l in recorder.get('train_loss')], label='train')
        axis.plot([l.mean().item() for l in recorder.get('test_loss')], label='test')
        axis.legend()
        axis.set_xlabel('iterations')
        axis.set_ylabel('loss')
        axis.grid()
        axis.set_title('loss')

        axis = axes_row1[2]
        axis.plot(recorder.get('train_acc'), label='train')
        axis.plot(recorder.get('test_acc'), label='test')
        axis.legend()
        axis.set_xlabel('iterations')
        axis.set_ylabel('accuracy')
        axis.grid()
        axis.set_title('accuracy')

        ## analysis
        axs = subfigs_rows[1].subplots(ncols=len(usvs), nrows=len(evals_i)+2)

        for i, (u, s, v) in enumerate(usvs):

            axis = axs[0, i]
            axis.set_title(f'iteration = {ntks["iterations"][i]}, train acc = {ntks["train_acc"][i]:.2f}')

            custom_imshow(axis, ntks['f_hat_test'][i], center=True)
            axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
            axis.set_xticks([]); axis.set_yticks([])

            axis = axs[1, i]
            custom_imshow(axis, torch.norm(ntks['jacobian'][i], dim=1), center=True)
            axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
            axis.set_xticks([]); axis.set_yticks([])

            for j, e_index in enumerate(evals_i):
                axis = axs[j+2, i]

                custom_imshow(axis, v[:, e_index], center=True)
                axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
                axis.annotate(f'$\lambda_{{ {e_index} }}$ = {s[e_index].item():.2e}', xycoords='axes fraction', xy=(.1, .9), bbox=dict(boxstyle="round", fc="w"))
                axis.set_xticks([]); axis.set_yticks([])

        axs[0, 0].set_ylabel(f'prediction')
        axs[1, 0].set_ylabel(f'jacobian norm')


        plt.savefig(f'plot_{args.dataset}.pdf')
        plt.show()

        mult = 3.5
        evals_i = [0, 1, 10, 100]

        fig = plt.figure(figsize=(17, 22), constrained_layout=True)
        subfigs_rows = fig.subfigures(2, 1, height_ratios=[1, 4])
        axes_row1 = subfigs_rows[0].subplots(1, 3)

        fig.suptitle(args.dataset)

        points = [(7, 7),
                (15, 15),
                (22, 22),
                (30, 30)]

        ## dataset
        axis = axes_row1[0]
        custom_imshow(axis, y_test)
        axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title('true distrib + dataset')

        ## training
        axis = axes_row1[1]

        axis.plot([l.mean().item() for l in recorder.get('train_loss')], label='train')
        axis.plot([l.mean().item() for l in recorder.get('test_loss')], label='test')
        axis.legend()
        axis.set_xlabel('iterations')
        axis.set_ylabel('loss')
        axis.grid()
        axis.set_title('loss')

        axis = axes_row1[2]
        axis.plot(recorder.get('train_acc'), label='train')
        axis.plot(recorder.get('test_acc'), label='test')
        axis.legend()
        axis.set_xlabel('iterations')
        axis.set_ylabel('accuracy')
        axis.grid()
        axis.set_title('accuracy')

        ## analysis
        axs = subfigs_rows[1].subplots(ncols=len(usvs), nrows=len(evals_i)+2)

        for i, (u, s, v) in enumerate(usvs):

            axis = axs[0, i]
            axis.set_title(f'iteration = {ntks["iterations"][i]}, train acc = {ntks["train_acc"][i]:.2f}')

            custom_imshow(axis, ntks['f_hat_test'][i], center=True)
            axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
            axis.set_xticks([]); axis.set_yticks([])

            axis = axs[1, i]
            custom_imshow(axis, torch.norm(ntks['jacobian'][i], dim=1), center=True)
            axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
            axis.set_xticks([]); axis.set_yticks([])

            jacs = ntks['jacobian'][i]
            jacs_mat = jacs.view(args.test_resolution, args.test_resolution, -1)

            for j, p_xy in enumerate(points):
                axis = axs[j+2, i]

                jac_this = jacs_mat[p_xy[0], p_xy[1], :]
                k_xy = torch.mv(jacs, jac_this)

                custom_imshow(axis, k_xy, center=True)
                axis.scatter(x_train[:, 0].cpu(), x_train[:, 1].cpu(), c=y_train.cpu(), marker='x', cmap=cmap)
                axis.set_xticks([]); axis.set_yticks([])
                axis.scatter([p_xy[0] / 30 - 1], [p_xy[1] / 30 - 1], c='black', s=25)

        axs[0, 0].set_ylabel(f'prediction')
        axs[1, 0].set_ylabel(f'jacobian norm')
        for j in range(len(points)):
            axs[j+2, 0].set_ylabel('k(x,•)')


        plt.savefig(f'plot_{args.dataset}.pdf')
        plt.show()
# %%

        mean_train_loss  = [l.mean().item() for l in recorder.get('train_loss')]

        # %%
        dists = (x_train[:, 0]**2 + x_train[:, 1]**2)**.5 - r
        cmap = plt.get_cmap('Spectral')

        def get_color(dists, i):
            mmax = torch.abs(dists).max()
            return cmap((dists[i] / mmax + .5).item())


        for i in range(len(dists)):
            plt.plot(mean_train_loss, [l[i] for l in recorder.get('train_loss')],
                    label='train',
                    c=get_color(dists, i))
        plt.grid()
        plt.xlim(plt.xlim()[1], 0)
        plt.ylim(0, 1)
        plt.xlabel('mean training loss')
        plt.ylabel('indiv. examples loss')
        plt.savefig(f'figures/seed_{args.seed}_alpha_{args.alpha}.pdf')
        plt.show()

        # %%
        dists = (x_train[:, 0]**2 + x_train[:, 1]**2)**.5 - r
        ranks = dists.argsort()
        indices = []
        binned_mean_loss = []
        binned_mean_dist = []
        for b in range(10):
            indices.append(ranks[b*10:(b+1)*10])
            binned_mean_loss.append([l[indices[-1]].mean().item()
                            for l in recorder.get('train_loss')])
            binned_mean_dist.append(dists[indices[-1]].mean())

        cmap = plt.get_cmap('Spectral')

        def get_color(dists, i):
            mmax = torch.abs(dists).max()
            return cmap((dists[i] / mmax + .5).item())

        for i in range(10):
            plt.plot(mean_train_loss, binned_mean_loss[i],
                    c=cmap(i/10),
                    label=str(binned_mean_dist[i].item()))
        plt.grid()
        plt.xlim(plt.xlim()[1], 0)
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('mean training loss')
        plt.ylabel('indiv. examples loss')
        plt.savefig(f'figures/bins_seed_{args.seed}_alpha_{args.alpha}.pdf')
        plt.show()

    return stats
# %%
