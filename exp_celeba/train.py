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
from datasets import CelebA
import torchvision.datasets.utils as dataset_utils
import os
from torch.utils.data import TensorDataset
from nngeometry.layers import WeightNorm2d

import copy

import pickle as pkl

def makedir_lazy(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%

data_path = '/Tmp/slurm.1459169.0/data'

# %% 

def get_celeba_stats(split):
  ds = CelebA(data_path=data_path, split=split)
  attrs = []
  dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
  for i, x, y, g in iter(dl):

    attrs.append([i.item(), y.item(), g.item()])

  return np.array(attrs)

if False:
    stats = get_celeba_stats('te')

    print(f'#blond persons: {(stats[:, 1] == 1).sum()}')
    print(f'#not blond persons: {(stats[:, 1] == 0).sum()}')
    print(f'#men: {(stats[:, 2] == 1).sum()}')
    print(f'#blond men: {((stats[:, 2] == 1) & (stats[:, 1] == 1)).sum()}')
    print(f'#women: {(stats[:, 2] == 0).sum()}')
    print(f'#blond women: {((stats[:, 2] == 0) & (stats[:, 1] == 1)).sum()}')

## TEST
#blond persons: 2660
#not blond persons: 17302
#men: 7715
#blond men: 180
#women: 12247
#blond women: 2480

## TRAIN
#blond persons: 24267
#not blond persons: 138503
#men: 68261
#blond men: 1387
#women: 94509
#blond women: 22880

# %%

def get_balanced_dataset(split, n):
    ds = CelebA(data_path=data_path, split=split)
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
  ds = CelebA(data_path=data_path, split=split)
  xs, ys, gs = [], [], []
  dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
  for i, x, y, g in iter(dl):

    xs.append(x.to('cuda'))
    ys.append(y.to('cuda'))
    gs.append(g.to('cuda'))

    if len(xs) >= 20000:
      break

  return TensorDataset(torch.cat(xs), torch.cat(ys), torch.cat(gs))

celeba_train_ds = get_celeba_gpu('tr')
celeba_train_balanced_ds = get_balanced_dataset('tr', 150)

# %% 

dl = torch.utils.data.DataLoader(celeba_train_ds, batch_size=1, shuffle=False)
for i, (x, y, g) in enumerate(iter(dl)):
  plt.figure()
  plt.imshow(x.cpu()[0].permute(1, 2, 0))
  plt.title(f'y={y.item()}, g={g.item()}')
  plt.show()
  if i > 3:
    break

# %% [markdown]
# ## Define neural network
# 
# The paper uses an MLP but a Convnet works fine too.

# %%


# %% [markdown]
# ## Test ERM as a baseline
# 
# Using ERM as a baseline, we expect to train a neural network that uses color instead of the actual digit to classify, completely failing on the test set when the colors are switched.

# %%
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

# %%
def test_model(model, model_0, alpha, device, test_loader, set_name="test set"):
    model.eval()
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

def loss_acc_by_group(output, target, gender):
    #gender: 1=man, 0=woman
    blond_and_man = gender #* target
    blond_and_woman = (1 - gender) #* target
    output = output[:, 0]
    loss_indivs = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    loss_flipped = (loss_indivs * blond_and_man).sum() / blond_and_man.sum() 
    loss_unflipped = (loss_indivs * blond_and_woman).sum() / blond_and_woman.sum() 

    acc_indivs = (output > 0) == target
    acc_flipped = (acc_indivs * blond_and_man).sum() / blond_and_man.sum() 
    acc_unflipped = (acc_indivs * blond_and_woman).sum() / blond_and_woman.sum() 

    return loss_indivs.mean(), loss_flipped, loss_unflipped, acc_indivs.float().mean(), acc_flipped, acc_unflipped


def erm_train(model, model_0, alpha, device, train_loader, train_balanced_loader,
        test_loader, optimizer, epoch, recorder):
    model.train()
    do_stop = 3
    for batch_idx, (data, target, gender) in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.no_grad():
            model_0_output = model_0(data)
        output = alpha * (model(data) - model_0_output)
        loss, loss_flipped, loss_unflipped, acc, acc_flipped, acc_unflipped = \
            loss_acc_by_group(output, target.float(), gender)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ({:.6f}, {:.6f})'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), loss_flipped.item(),
                        loss_unflipped.item()))
            recorder.save('loss', loss.item())
            recorder.save('acc', acc.item())
            recorder.save('loss_flipped', loss_flipped.item())
            recorder.save('loss_unflipped', loss_unflipped.item())
            recorder.save('acc_flipped', acc_flipped.item())
            recorder.save('acc_unflipped', acc_unflipped.item())

            with torch.no_grad():
                x_test, y_test, flip_test = next(iter(test_loader))
                y_test = y_test.to(device).float()
                output_test = alpha * (model(x_test) - model_0(x_test))
                (loss_test, loss_flipped_test, loss_unflipped_test, acc_test,
                    acc_flipped_test, acc_unflipped_test) = \
                    loss_acc_by_group(output_test, y_test, flip_test)

            recorder.save('loss_test', loss_test.item())
            recorder.save('acc_test', acc_test.item())
            recorder.save('loss_flipped_test', loss_flipped_test.item())
            recorder.save('loss_unflipped_test', loss_unflipped_test.item())
            recorder.save('acc_flipped_test', acc_flipped_test.item())
            recorder.save('acc_unflipped_test', acc_unflipped_test.item())

            with torch.no_grad():
                x_test, y_test, flip_test = next(iter(train_balanced_loader))
                y_test = y_test.to(device).float()
                output_test = alpha * (model(x_test) - model_0(x_test))
                (loss_test, loss_flipped_test, loss_unflipped_test, acc_test,
                    acc_flipped_test, acc_unflipped_test) = \
                    loss_acc_by_group(output_test, y_test, flip_test)

            recorder.save('loss_trainb', loss_test.item())
            recorder.save('acc_trainb', acc_test.item())
            recorder.save('loss_flipped_trainb', loss_flipped_test.item())
            recorder.save('loss_unflipped_trainb', loss_unflipped_test.item())
            recorder.save('acc_flipped_trainb', acc_flipped_test.item())
            recorder.save('acc_unflipped_trainb', acc_unflipped_test.item())
        # if torch.isnan(loss) or loss.cpu() < .08:
        if torch.isnan(loss) or acc.cpu() == 1:
            do_stop -= 1
            if do_stop == 0:
                return True
        else:
            do_stop = 3
    return False


def train_and_test_erm(alpha, model, model_0, all_train_loader, train_balanced_loader,
        test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=0.01 / alpha**2, momentum=.9)
    recorder = Recorder()

    for epoch in range(1, 250):
        do_stop = erm_train(model, model_0, alpha, device, all_train_loader, train_balanced_loader, test_loader, optimizer, epoch, recorder)

        # train_loss, train_acc = test_model(model, model_0, alpha, device, all_train_loader, set_name='train set')
        # test_loss, test_acc = test_model(model, model_0, alpha, device, test_loader)

        if do_stop:
            break
    return recorder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_loader = torch.utils.data.DataLoader(celeba_train_ds, batch_size=100,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(celeba_test_ds, batch_size=len(celeba_test_ds),
                                           shuffle=False)
train_balanced_loader = torch.utils.data.DataLoader(
    celeba_train_balanced_ds, batch_size=len(celeba_train_balanced_ds), shuffle=False)
model = models.resnet.resnet18(norm_layer=torch.nn.Identity)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model = model.cuda()

# alphas = 10**np.arange(0, 3.5, 1)
alphas = [0.5, 1, 2]
recorders = []

model_0 = copy.deepcopy(model)

for alpha in alphas:
    print(f'------ alpha = {alpha}')
    model_copy = copy.deepcopy(model)
    recorder = train_and_test_erm(alpha, model_copy, model_0, train_loader, train_balanced_loader, test_loader, device)
    recorders.append(recorder)

# %%

pkl.dump(recorders, open('./recorders.pkl', 'wb'))

# %%
alphas = [0.5, 1, 2]
recorders = pkl.load(open('recorders.pkl', 'rb'))

# %%
N = 25
smoothen_moving_average = lambda x: np.convolve(x, np.ones(N)/N, mode='valid')

def smoothen_running_average(x):
  gamma = .9
  o = []
  ra = x[0]
  for xi in x:
    ra = gamma * ra + (1 - gamma) * xi
    o.append(ra)
  return np.array(o)

no_smoothing = lambda x: x

smoothen = smoothen_moving_average

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

cmaps = ['spring', 'summer', 'winter']

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
