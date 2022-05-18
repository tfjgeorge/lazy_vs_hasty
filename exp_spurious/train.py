# %%
import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms, models
# from torchvision import datasets
from datasets import CelebA, Waterbirds
import logging
logging.basicConfig(level=logging.INFO)
import torchvision.datasets.utils as dataset_utils
import os
from torch.utils.data import TensorDataset
from nngeometry.layers import WeightNorm2d
from pytorch_memlab import MemReporter
import argparse

import copy

import pickle as pkl

import sys
sys.path.append('..')
from plot_utils import *
from linearization_utils import LinearizationProbe, ModelLinearKnob
from train_utils import Recorder, ProbeAssistant

# %%

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# CelebA:

ds_suffix = 'celeba'
DS = CelebA

# Waterbirds:

ds_suffix = 'waterbirds'
DS = Waterbirds

##

if 'SLURM_TMPDIR' in os.environ:
    slurm_tmpdir = os.environ['SLURM_TMPDIR']
else:
    slurm_tmpdir = '/Tmp/slurm.1836774.0'
save_dir = f'/network/projects/g/georgeth/linvsnonlin/{ds_suffix}'

data_path = os.path.join(slurm_tmpdir, 'data')

PATIENCE_MAX = 12


# %%

def get_celeba_stats(split):
  ds = DS(data_path=data_path, split=split)
  attrs = []
  dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
  for i, x, y, g in iter(dl):

    attrs.append([i.item(), y.item(), g.item()])

  return np.array(attrs)

if False:
    for ds in ['train', 'test']:

        print(f'{ds} dataset')
        stats = get_celeba_stats(ds[:2])

        print(f'#blond persons: {(stats[:, 1] == 1).sum()}')
        print(f'#not blond persons: {(stats[:, 1] == 0).sum()}')
        print(f'#men: {(stats[:, 2] == 1).sum()}')
        print(f'#blond men: {((stats[:, 2] == 1) & (stats[:, 1] == 1)).sum()}')
        print(f'#women: {(stats[:, 2] == 0).sum()}')
        print(f'#blond women: {((stats[:, 2] == 0) & (stats[:, 1] == 1)).sum()}')

## CelebA TEST
#blond persons: 2660
#not blond persons: 17302
#men: 7715
#blond men: 180
#women: 12247
#blond women: 2480

## CelebA TRAIN
#blond persons: 24267
#not blond persons: 138503
#men: 68261
#blond men: 1387
#women: 94509
#blond women: 22880

## Waterbirds TRAIN
#blond persons: 1113
#not blond persons: 3682
#men: 1241
#blond men: 1057
#women: 3554
#blond women: 56

## Waterbirds TEST
#blond persons: 1284
#not blond persons: 4510
#men: 2897
#blond men: 642
#women: 2897
#blond women: 642

# %%

def get_balanced_dataset(split, n):
    ds = DS(data_path=data_path, split=split)
    attrs = []
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    blond_men = []
    notblond_men = []
    blond_women = []
    notblond_women = []
    for i, x, y, g in iter(dl):
        if y == 0 and g == 0 and (len(notblond_women) < n):
            notblond_women.append((x.to('cuda'), y.to('cuda'), g.to('cuda')))
        if y == 1 and g == 0 and (len(blond_women) < n):
            blond_women.append((x.to('cuda'), y.to('cuda'), g.to('cuda')))
        if y == 0 and g == 1 and (len(notblond_men) < n):
            notblond_men.append((x.to('cuda'), y.to('cuda'), g.to('cuda')))
        if y == 1 and g == 1 and (len(blond_men) < n):
            blond_men.append((x.to('cuda'), y.to('cuda'), g.to('cuda')))

        if (len(blond_men) == n and len(notblond_men) == n and
            len(blond_women) == n and len(notblond_women) == n):
            break

    xs = torch.cat([torch.cat([d[0] for d in blond_men]),
                      torch.cat([d[0] for d in notblond_men]),
                      torch.cat([d[0] for d in blond_women]),
                      torch.cat([d[0] for d in notblond_women])])
    ys = torch.cat([torch.cat([d[1] for d in blond_men]),
                      torch.cat([d[1] for d in notblond_men]),
                      torch.cat([d[1] for d in blond_women]),
                      torch.cat([d[1] for d in notblond_women])])
    gs = torch.cat([torch.cat([d[2] for d in blond_men]),
                      torch.cat([d[2] for d in notblond_men]),
                      torch.cat([d[2] for d in blond_women]),
                      torch.cat([d[2] for d in notblond_women])])
    return TensorDataset(xs, ys, gs)

celeba_test_ds = get_balanced_dataset('te', 150)

# %%

def get_celeba_gpu(split):
  ds = DS(data_path=data_path, split=split)
  xs, ys, gs = [], [], []
  dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
  for i, x, y, g in iter(dl):

    xs.append(x.to('cuda'))
    ys.append(y.to('cuda'))
    gs.append(g.to('cuda'))

    if len(xs) >= 15000:
      break

  return TensorDataset(torch.cat(xs), torch.cat(ys), torch.cat(gs))

logging.info(f"Loading {ds_suffix} to GPU")
celeba_train_ds = get_celeba_gpu('tr')
celeba_train_balanced_ds = get_balanced_dataset('tr', 150)

# %%

def loss_acc_by_group_dl(model_linear_knob, loader):
    loss_s = 0
    loss_act_s = 0 # actual
    loss_spu_s = 0 # spurious
    acc_s = 0
    acc_act_s = 0
    acc_spu_s = 0
    count = 0
    count_act = 0
    count_spu = 0
    with torch.no_grad():
        for x, y, flip in iter(loader):
            y = y.to(device).float()
            output = model_linear_knob.pred_nograd(x)
            (loss, loss_act, loss_spu,
             acc, acc_act, acc_spu,
             c, c_act, c_spu) = \
                loss_acc_by_group(output, y, flip, reduce=False)
            loss_s += loss
            loss_act_s += loss_act
            loss_spu_s += loss_spu
            acc_s += acc
            acc_act_s += acc_act
            acc_spu_s += acc_spu
            count += c
            count_act += c_act
            count_spu += c_spu
    return (loss_s / count, loss_act_s / count_act, loss_spu_s / count_spu,
            acc_s / count, acc_act_s / count_act, acc_spu_s / count_spu)


# %%
def test_model(model, model_0, alpha, device, test_loader, set_name="test set"):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, flip_color in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = alpha * (model(data) - model_0(data))
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                                torch.Tensor([1.0]).to(device),
                                torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        set_name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)

def loss_acc_by_group(output, target, gender, reduce=True):
    #gender: 1=man, 0=woman
    if ds_suffix == 'celeba':
        switch_min = gender #* target
        switch_maj = (1 - gender) #* target
    elif ds_suffix == 'waterbirds':
        switch_min = gender * (1 - target) + (1 - gender) * target
        switch_maj = 1 - switch_min

    count_min = switch_min.sum()
    count_maj = switch_maj.sum()

    output = output[:, 0]
    loss_indivs = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    loss_min = (loss_indivs * switch_min).sum()
    loss_maj = (loss_indivs * switch_maj).sum()

    acc_indivs = (output > 0) == target
    acc_min = (acc_indivs * switch_min).sum()
    acc_maj = (acc_indivs * switch_maj).sum()

    if reduce:
        return (loss_indivs.mean(), loss_min / count_min, loss_maj / count_maj,
                acc_indivs.float().mean(), acc_min / count_min, acc_maj / count_maj)
    else:
        return (loss_indivs.sum(), loss_min, loss_maj,
                acc_indivs.float().sum(), acc_min, acc_maj,
                loss_indivs.size(0), count_min, count_maj)

def erm_train(model_linear_knob, device, train_loader, train_balanced_loader,
        test_loader, optimizer, epoch, recorder, linprobe, probe_assistant):
    do_stop = PATIENCE_MAX

    for batch_idx, (x, target, gender) in enumerate(train_loader):

        # for measurement only (update is done below)
        optimizer.zero_grad()
        output = model_linear_knob.pred_nograd(x)
        loss, loss_flipped, loss_unflipped, acc, acc_flipped, acc_unflipped = \
            loss_acc_by_group(output, target.float(), gender)

        probe_assistant.record_loss(loss.item())
        if probe_assistant.do_probe() or batch_idx % 100 == 0:
            recorder.save('loss', loss.item())
            recorder.save('acc', acc.item())
            recorder.save('loss_flipped', loss_flipped.item())
            recorder.save('loss_unflipped', loss_unflipped.item())
            recorder.save('acc_flipped', acc_flipped.item())
            recorder.save('acc_unflipped', acc_unflipped.item())

            (loss_test, loss_flipped_test, loss_unflipped_test, acc_test,
                    acc_flipped_test, acc_unflipped_test) = loss_acc_by_group_dl(model_linear_knob, test_loader)

            recorder.save('loss_test', loss_test.item())
            recorder.save('acc_test', acc_test.item())
            recorder.save('loss_flipped_test', loss_flipped_test.item())
            recorder.save('loss_unflipped_test', loss_unflipped_test.item())
            recorder.save('acc_flipped_test', acc_flipped_test.item())
            recorder.save('acc_unflipped_test', acc_unflipped_test.item())

            (loss_test, loss_flipped_test, loss_unflipped_test, acc_test,
                    acc_flipped_test, acc_unflipped_test) = loss_acc_by_group_dl(model_linear_knob, train_balanced_loader)

            recorder.save('loss_trainb', loss_test.item())
            recorder.save('acc_trainb', acc_test.item())
            recorder.save('loss_flipped_trainb', loss_flipped_test.item())
            recorder.save('loss_unflipped_trainb', loss_unflipped_test.item())
            recorder.save('acc_flipped_trainb', acc_flipped_test.item())
            recorder.save('acc_unflipped_trainb', acc_unflipped_test.item())

            if recorder.len('loss') % 5 == 1:
                recorder.save('loss_100', loss.item())
                recorder.save('sign_similarity',
                              linprobe.sign_similarity(linprobe.get_signs(),
                                                       linprobe.buffer['signs0']).item())
                recorder.save('ntk_alignment',
                              linprobe.kernel_alignment(linprobe.get_ntk(),
                                                        linprobe.buffer['ntk0']).item())
                recorder.save('repr_alignment',
                              linprobe.representation_alignment(linprobe.get_last_layer_representation(),
                                                                linprobe.buffer['repr0']).item())

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({:.6f}, {:.6f})'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), loss_flipped.item(),
                        loss_unflipped.item()))
        if torch.isnan(loss) or acc.cpu() == 1:
            do_stop -= 1
            print(f'patience: {do_stop}')
            if do_stop == 0:
                return True
        else:
            do_stop = PATIENCE_MAX

        # update
        optimizer.zero_grad()
        output = model_linear_knob.pred(x)
        loss, loss_flipped, loss_unflipped, acc, acc_flipped, acc_unflipped = \
            loss_acc_by_group(output, target.float(), gender)
        loss.backward()
        optimizer.step()

    return False


def train_and_test_erm(alpha, model, all_train_loader, train_balanced_loader,
        test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=0.001 / alpha**2, momentum=.9)
    recorder = Recorder()

    model_linear_knob = ModelLinearKnob(model, copy.deepcopy(model), alpha)
    probe_assistant = ProbeAssistant(np.log(2), .96)

    linprobe = LinearizationProbe(model, test_loader)
    linprobe.buffer['signs0'] = linprobe.get_signs().detach()
    linprobe.buffer['ntk0'] = linprobe.get_ntk().detach()
    linprobe.buffer['repr0'] = linprobe.get_last_layer_representation().detach()

    for epoch in range(1, 750):
        do_stop = erm_train(model_linear_knob, device, all_train_loader,
                            train_balanced_loader, test_loader, optimizer, epoch, recorder,
                            linprobe, probe_assistant)

        if do_stop:
            break
    return recorder

def main(pkl_path):
    train_loader = torch.utils.data.DataLoader(celeba_train_ds, batch_size=100,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(celeba_test_ds, batch_size=150,
                                            shuffle=False)
    train_balanced_loader = torch.utils.data.DataLoader(
        celeba_train_balanced_ds, batch_size=len(celeba_train_balanced_ds), shuffle=False)
    if ds_suffix == 'celeba':
        model = models.resnet.resnet18(norm_layer=torch.nn.Identity)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.train()
    elif ds_suffix == 'waterbirds':
        model = models.resnet.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.eval()
    model = model.cuda()

    # alphas = 10**np.arange(0, 3.5, 1)
    alphas = [0.5, 1, 100]
    recorders = []

    for alpha in alphas:
        logging.info(f"Starting training for alpha = {alpha}")
        model_copy = copy.deepcopy(model)
        recorder = train_and_test_erm(alpha, model_copy, train_loader, train_balanced_loader, test_loader, device)
        recorders.append(recorder)

    pkl.dump(recorders, open(pkl_path, 'wb'))

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save', default=None, type=str, help='Path to store results')
    args = parser.parse_args()

    main(args.save)
    exit()

# %%
# alphas = [0.5, 1, 10]
# recorders = pkl.load(open(pkl_path, 'rb'))

# %%
def smoothen_moving_average(N):
    return lambda x: np.convolve(x, np.ones(N)/N, mode='valid')

def smoothen_running_average(x):
  gamma = .9
  o = []
  ra = x[0]
  for xi in x:
    ra = gamma * ra + (1 - gamma) * xi
    o.append(ra)
  return np.array(o)

no_smoothing = lambda x: x

smoothen = smoothen_running_average

# %%
plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    ax = plt.plot(smoothen(r.get('loss_flipped_trainb')), label=f'alpha={alpha}')
    plt.plot(smoothen(r.get('loss_unflipped_trainb')), color=ax[0].get_color())
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
# plt.xlim(5e-1, 3)
plt.ylim(0, 3)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.show()

# %%
plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    ax = plt.plot(smoothen(r.get('acc_flipped')), label=f'alpha={alpha}')
    plt.plot(smoothen(r.get('acc_unflipped')), color=ax[0].get_color())
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend()
# plt.xlim(5e-1, 3)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.show()

# %%

plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    ax = plt.plot(smoothen(r.get('sign_similarity')), label=f'alpha={alpha}')
plt.xlabel('iterations')
plt.ylabel('sign_similarity')
plt.legend()
# plt.xlim(5e-1, 3)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.show()

# %%

plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    ax = plt.plot(smoothen(r.get('loss_100')), smoothen(r.get('sign_similarity')), label=f'alpha={alpha}')
plt.xlabel('train loss')
plt.ylabel('sign_similarity')
plt.legend()

xlim = plt.xlim()
plt.xlim(xlim[1], xlim[0])
# plt.xlim(5e-1, 3)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.savefig(f'figures/signsim_vs_loss.pdf')
plt.show()

# %%

for prefix in ['trainb', 'test']:
    plt.figure(figsize=(10, 10))

    for r, alpha in zip(recorders, alphas):
        plt.scatter(smoothen(r.get(f'loss_unflipped_{prefix}')), smoothen(r.get(f'loss_flipped_{prefix}')), label=f'alpha={alpha}', marker='x', s=1)
    plt.ylabel(f'{prefix} loss for blond and man')
    plt.xlabel(f'{prefix} loss for blond and woman')
    plt.legend()
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    plt.savefig(f'figures/{prefix}_loss_spur_vs_actual.pdf')
    plt.show()


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
    x, y = rotate(x, y, np.pi/4, (.5, .5))
    x, y = smoothen_lowess(x, y)
    x, y = rotate_back(x, y, np.pi/4, (.5, .5))
    return x, y


for prefix in ['trainb', 'test']:
    plt.figure(figsize=(10, 10))

    for r, alpha in zip(recorders, alphas):
        x, y = smoothen_xy(r.get(f'acc_unflipped_{prefix}'),
                           r.get(f'acc_flipped_{prefix}'))
        plt.plot(x, y, label=f'alpha={alpha}')
    plt.ylabel(f'{prefix} acc blond and man')
    plt.xlabel(f'{prefix} acc blond and woman')
    plt.legend()
    # plt.ylim(1e-1, 3)
    # plt.xlim(1e-1, 1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    plt.savefig(f'figures/{prefix}_acc_spur_vs_actual.pdf')
    plt.show()

# %%

for prefix in ['trainb', 'test']:
    plt.figure(figsize=(10, 10))

    for r, alpha in zip(recorders, alphas):
        plt.scatter(smoothen(r.get(f'acc_unflipped_{prefix}')),
                    smoothen(r.get(f'acc_flipped_{prefix}')),
                    label=f'alpha={alpha}',
                    marker='x', s=1)
    plt.ylabel(f'{prefix} acc blond and man')
    plt.xlabel(f'{prefix} acc blond and woman')
    plt.legend()
    # plt.ylim(1e-1, 3)
    # plt.xlim(1e-1, 1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    plt.savefig(f'figures/{prefix}_acc_spur_vs_actual.pdf')
    plt.show()

# %%

cmaps = ['spring', 'summer', 'winter', 'winter']

for prefix in ['trainb', 'test']:
    plt.figure(figsize=(10, 10))

    for i, (r, alpha) in enumerate(zip(recorders, alphas)):
        x = smoothen(r.get(f'acc_unflipped_{prefix}'))
        y = smoothen(r.get(f'acc_flipped_{prefix}'))
        plt.scatter(x, y,
                 label=f'alpha={alpha}',
                 marker='x', s=10,
                 cmap=cmaps[i], c=np.linspace(0, 1, len(x)))
    plt.ylabel(f'{prefix} acc blond and man')
    plt.xlabel(f'{prefix} acc blond and woman')
    plt.legend()
    # plt.ylim(1e-1, 3)
    # plt.xlim(1e-1, 1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    plt.savefig(f'figures/{prefix}_acc_spur_vs_actual.pdf')
    plt.show()

# %%
plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    plt.scatter(smoothen(r.get('loss')), smoothen(r.get('loss_test')), label=f'alpha={alpha}', marker='x', s=1)
plt.xlabel('loss train')
plt.ylabel('loss test')
plt.legend()
# plt.ylim(1e-1, 3)
# plt.xlim(1e-1, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.savefig('figures/loss_test_vs_train.pdf')
plt.show()

# %%
plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    plt.scatter(smoothen(r.get('acc')), smoothen(r.get('acc_test')), label=f'alpha={alpha}', marker='x', s=1)
plt.xlabel('acc train')
plt.ylabel('acc test')
plt.legend()
# plt.ylim(1e-1, 3)
# plt.xlim(1e-1, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.savefig('figures/acc_test_vs_train.pdf')
plt.show()
# %%