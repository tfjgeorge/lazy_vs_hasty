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
from train_utils import Recorder, ProbeAssistant, InfiniteDataLoader
from file_utils import resolve_tmpdir

from torch.utils.data import DataLoader

# %%

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

def get_balanced_dataset(split, n, DS):
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

# %%

def get_celeba_gpu(split, DS):
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
    if args.task == 'celeba':
        switch_min = gender #* target
        switch_maj = (1 - gender) #* target
    elif args.task == 'waterbirds':
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


def train_loop(model_linear_knob, train_loader, optimizer, max_epochs,
               recorder, probe_assistant):
    stop_patience = PATIENCE_MAX

    for batch_idx, epoch, (x, target, gender) in train_loader:

        optimizer.zero_grad()
        probe_assistant.step(force_probe=(batch_idx==0))

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{} examples]\tLoss: {:.6f} ({:.6f}, {:.6f})'.format(
                epoch, batch_idx * len(x),
                    recorder.get('loss')[-1],
                    recorder.get('loss_flipped')[-1],
                    recorder.get('loss_unflipped')[-1]))

        if batch_idx > 0 and acc.cpu() == 1:
            stop_patience -= 1
            print(f'patience: {stop_patience}')
            if stop_patience == 0:
                break
        else:
            stop_patience = PATIENCE_MAX
        if epoch >= max_epochs or \
            (batch_idx > 0 and torch.isnan(loss)): break

        # update
        optimizer.zero_grad()
        output = model_linear_knob.pred(x)
        loss, loss_flipped, loss_unflipped, acc, acc_flipped, acc_unflipped = \
            loss_acc_by_group(output, target.float(), gender)
        loss.backward()
        optimizer.step()

        probe_assistant.record_loss(loss.item())

    probe_assistant.step(force_probe=True)


def evaluate_losses(recorder, model_linear_knob, linear_probe, loaders):
    def _evaluate_losses():
        (loss_train, loss_flipped_train, loss_unflipped_train, acc_train,
                acc_flipped_train, acc_unflipped_train) = \
                    loss_acc_by_group_dl(model_linear_knob, loaders['train'])

        recorder.save('loss', loss_train.item())
        recorder.save('acc', acc_train.item())
        recorder.save('loss_flipped', loss_flipped_train.item())
        recorder.save('loss_unflipped', loss_unflipped_train.item())
        recorder.save('acc_flipped', acc_flipped_train.item())
        recorder.save('acc_unflipped', acc_unflipped_train.item())

        (loss_test, loss_flipped_test, loss_unflipped_test, acc_test,
                acc_flipped_test, acc_unflipped_test) = \
                    loss_acc_by_group_dl(model_linear_knob, loaders['test'])

        recorder.save('loss_test', loss_test.item())
        recorder.save('acc_test', acc_test.item())
        recorder.save('loss_flipped_test', loss_flipped_test.item())
        recorder.save('loss_unflipped_test', loss_unflipped_test.item())
        recorder.save('acc_flipped_test', acc_flipped_test.item())
        recorder.save('acc_unflipped_test', acc_unflipped_test.item())

        (loss_trainb, loss_flipped_trainb, loss_unflipped_trainb, acc_trainb,
                acc_flipped_trainb, acc_unflipped_trainb) = \
                    loss_acc_by_group_dl(model_linear_knob, loaders['trainb'])

        recorder.save('loss_trainb', loss_trainb.item())
        recorder.save('acc_trainb', acc_trainb.item())
        recorder.save('loss_flipped_trainb', loss_flipped_trainb.item())
        recorder.save('loss_unflipped_trainb', loss_unflipped_trainb.item())
        recorder.save('acc_flipped_trainb', acc_flipped_trainb.item())
        recorder.save('acc_unflipped_trainb', acc_unflipped_trainb.item())

        if recorder.len('loss') % 5 == 1:
            recorder.save('loss_100', loss_train.item())
            recorder.save('sign_similarity',
                            linear_probe.sign_similarity(
                                linear_probe.get_signs(),
                                linear_probe.buffer['signs0']).item())
            recorder.save('ntk_alignment',
                            linear_probe.kernel_alignment(
                                linear_probe.get_ntk(),
                                linear_probe.buffer['ntk0']).item())
            recorder.save('repr_alignment',
                            linear_probe.representation_alignment(
                                linear_probe.get_last_layer_representation(),
                                linear_probe.buffer['repr0']).item())
                                
    return _evaluate_losses                                                        


def train_alpha(alpha, model, train_loader, all_loaders):

    lr = 0.001 if args.task == 'waterbirds' else 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr / alpha**2, momentum=.9)
    recorder = Recorder()

    linprobe = LinearizationProbe(model, all_loaders['test'])
    linprobe.buffer['signs0'] = linprobe.get_signs().detach()
    linprobe.buffer['ntk0'] = linprobe.get_ntk().detach()
    linprobe.buffer['repr0'] = linprobe.get_last_layer_representation().detach()

    model_linear_knob = ModelLinearKnob(model, copy.deepcopy(model), alpha)
    probe_assistant = ProbeAssistant(evaluate_losses(recorder=recorder,
                                                     model_linear_knob=model_linear_knob,
                                                     linear_probe=linprobe,
                                                     loaders=all_loaders),
                                     np.log(2), .96)

    train_loop(model_linear_knob, train_loader, optimizer, 750, recorder,
               probe_assistant)

    return recorder


def main(pkl_path, task):
    if task == 'celeba':
        model = models.resnet.resnet18(norm_layer=torch.nn.Identity)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.train()
    elif task == 'waterbirds':
        model = models.resnet.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.eval()
    model = model.cuda()

    alphas = [0.5, 1, 100]
    recorders = []

    for alpha in alphas:
        logging.info(f"Starting training for alpha = {alpha}")

        # preparing dataloaders
        train_loader_inf = InfiniteDataLoader(celeba_train_ds, batch_size=100,
                                            shuffle=True)
        train_loader = DataLoader(celeba_train_ds, batch_size=100, shuffle=False)
        test_loader = DataLoader(celeba_test_ds, batch_size=150, shuffle=False)
        train_balanced_loader = DataLoader(celeba_train_balanced_ds,
                batch_size=len(celeba_train_balanced_ds), shuffle=False)
        all_loaders = {
            'train': train_loader,
            'test': test_loader,
            'trainb': train_balanced_loader
        }

        #copy initial model
        model_copy = copy.deepcopy(model)

        # start training, get recorder
        recorder = train_alpha(alpha, model_copy, train_loader_inf, all_loaders)
        recorders.append(recorder)

    pkl.dump(recorders, open(pkl_path, 'wb'))

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save', default=None, type=str, help='Path to store results')
    parser.add_argument('--task', default=None, type=str, choices=['celeba', 'waterbirds'],
                        help='Task')
    args = parser.parse_args()

    if args.task == 'celeba':
        DS = CelebA
    elif args.task == 'waterbirds':
        DS = Waterbirds

    tmp_dir = resolve_tmpdir()
    save_dir = f'/network/projects/g/georgeth/linvsnonlin/{args.task}'
    data_path = os.path.join(tmp_dir, 'data')

    logging.info(f"Loading {args.task} dataset to GPU")
    celeba_train_ds = get_celeba_gpu('tr', DS)
    celeba_train_balanced_ds = get_balanced_dataset('tr', 150, DS)
    celeba_test_ds = get_balanced_dataset('te', 150, DS)

    main(args.save, args.task)
