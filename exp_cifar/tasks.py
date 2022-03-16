import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, KMNIST
from models import VGG, resnet18
import random
import numpy as np

default_datapath = '/tmp/data'
if 'SLURM_TMPDIR' in os.environ:
    default_datapath = os.path.join(os.environ['SLURM_TMPDIR'], 'data')
elif 'SLURM_JOB_ID' in os.environ:
    default_datapath = os.path.join('/Tmp', f'slurm.{os.environ["SLURM_JOB_ID"]}.0', 'data')

def to_tensordataset(dataset, logits=None):
    d = next(iter(DataLoader(dataset,
                  batch_size=len(dataset))))
    if logits is None:
        logits = torch.zeros((len(dataset), d[1].max()+1), dtype=torch.float, device='cuda')
    return TensorDataset(d[0].to('cuda'), d[1].to('cuda'), logits)

def extract_small_loader(baseloader, length, batch_size):
    datas = []
    targets = []
    logits = []
    i = 0
    for d, t, l in iter(baseloader):
        datas.append(d.to('cuda'))
        targets.append(t.to('cuda'))
        logits.append(l.to('cuda'))
        i += d.size(0)
        if i >= length:
            break
    datas = torch.cat(datas)[:length]
    targets = torch.cat(targets)[:length]
    logits = torch.cat(logits)[:length]
    dataset = TensorDataset(datas.to('cuda'), targets.to('cuda'), logits)

    return DataLoader(dataset, shuffle=False, batch_size=batch_size)

def get_cifar10(args, sampler=None):
    trfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = Subset(CIFAR10(root=default_datapath, train=True, download=True,
                              transform=trfms),
                      range(40000))
    trainset = to_tensordataset(trainset)
    if sampler is None:
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 shuffle=True)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                sampler=sampler)
    trainloader_det = DataLoader(trainloader.dataset, batch_size=1000,
                                 shuffle=False)

    testset = Subset(CIFAR10(root=default_datapath, train=True, download=True,
                             transform=trfms),
                     range(40000, 50000))
    testloader = DataLoader(to_tensordataset(testset), batch_size=1000,
                            shuffle=False)

    return trainloader, trainloader_det, testloader

def get_mnist_normalization(args):
    trainset_mnist = MNIST(default_datapath, train=True, download=True)
    mean_mnist = (trainset_mnist.data.float() / 255).mean()
    std_mnist = (trainset_mnist.data.float() / 255).std()
    if args.diff == 0 or args.diff_type == 'random':
        return mean_mnist.item(), std_mnist.item()

    # otherwise we need to include kmnist before normalization
    trainset_kmnist = KMNIST(default_datapath, train=True, download=True)
    mean_kmnist = (trainset_kmnist.data.float() / 255).mean()
    std_kmnist = (trainset_kmnist.data.float() / 255).std()

    mean_both = args.diff * mean_kmnist + (1 - args.diff) * mean_mnist
    std_both = (args.diff * std_kmnist**2 + (1 - args.diff) * std_mnist**2) ** .5
    return mean_both.item(), std_both.item()

def get_mnist(args):
    mean, std = get_mnist_normalization(args)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    trainset = MNIST(root=default_datapath, train=True, download=True,
                     transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = MNIST(root=default_datapath, train=False, download=True,
                    transform=transform_train)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    return trainloader, testloader

def add_difficult_examples(dataloaders, args):
    # adds difficult examples and extract small
    # dataloaders
    if args.diff_type == 'random':
        trainset = dataloaders['train'].dataset
        x_easy = []
        y_easy = []
        logits_easy = []
        x_diff = []
        y_diff = []
        logits_diff = []
        targets = trainset.tensors[1]
        for i in range(len(targets)):
            if random.random() < args.diff:
                # choose a different target
                target = targets[i]
                while target == targets[i]:
                    target = random.randint(0, 9)
                targets[i] = target
                x_diff.append(trainset[i][0])
                y_diff.append(targets[i])
                logits_diff.append(trainset.tensors[2][i])
            else:
                x_easy.append(trainset[i][0])
                y_easy.append(targets[i])
                logits_easy.append(trainset.tensors[2][i])
        # print(x_easy)
        x_easy = torch.stack(x_easy)
        y_easy = torch.tensor(y_easy)
        logits_easy = torch.stack(logits_easy)
        x_diff = torch.stack(x_diff)
        y_diff = torch.tensor(y_diff)
        logits_diff = torch.stack(logits_diff)
    elif args.diff_type == 'other' and args.task[:5] == 'mnist':
        trainset = dataloaders['train'].dataset
        trainset_kmnist = KMNIST(default_datapath, train=True, download=True,
                                 transform=trainset.transform)
        mnist_len = len(trainset)
        kmnist_len = int(args.diff * mnist_len)
        indices = np.arange(len(trainset_kmnist))
        np.random.shuffle(indices)
        indices = indices[:kmnist_len]

        # apply transforms by hand
        x_easy = []
        y_easy = []
        x_diff = []
        y_diff = []
        for i in range(len(trainset.targets)):
            x_easy.append(trainset[i][0])
            y_easy.append(trainset.targets[i])
        for i in indices:
            x_diff.append(trainset_kmnist[i][0])
            y_diff.append(trainset_kmnist.targets[i])
        x_easy = torch.stack(x_easy)
        y_easy = torch.tensor(y_easy)
        x_diff = torch.stack(x_diff)
        y_diff = torch.tensor(y_diff)

        x = torch.cat([x_easy, x_diff])
        y = torch.cat([y_easy, y_diff])
        trainset_both = TensorDataset(x, y)
        dataloaders['train'] = DataLoader(trainset_both, batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError

    indices = np.arange(len(y_easy))
    np.random.shuffle(indices)
    indices = indices[:1000]
    x_easy = x_easy[indices]
    y_easy = y_easy[indices]
    logits_easy = logits_easy[indices]

    indices = np.arange(len(y_diff))
    np.random.shuffle(indices)
    indices = indices[:1000]
    x_diff = x_diff[indices]
    y_diff = y_diff[indices]
    logits_diff = logits_diff[indices]

    dataloaders['mini_train_easy'] = DataLoader(TensorDataset(x_easy.to('cuda'), y_easy.to('cuda'), logits_easy),
                                                batch_size=1000, shuffle=False)
    dataloaders['mini_train_diff'] = DataLoader(TensorDataset(x_diff.to('cuda'), y_diff.to('cuda'), logits_diff),
                                                batch_size=1000, shuffle=False)


def get_task(args, sampler=None):
    dataloaders = dict()

    task_name, model_name = args.task.split('_')

    if task_name == 'cifar10':
        if args.depth != 0:
            raise NotImplementedError
        dataloaders['train'], dataloaders['train_deterministic'], dataloaders['test'] = \
            get_cifar10(args, sampler=sampler)
        if model_name == 'vgg19':
            model = VGG('VGG19', base=args.width, bn=args.batch_norm)
        elif model_name == 'resnet18':
            model = resnet18()
            if args.width != 0:
                raise NotImplementedError
    elif task_name == 'mnist':
        dataloaders['train'], dataloaders['test'] = get_mnist(args)
        if model_name == 'fc':
            layers = [nn.Flatten(), nn.Linear(28 * 28, args.width), nn.ReLU()] + \
                     [nn.Linear(args.width, args.width), nn.ReLU()] * (args.depth - 2) + \
                     [nn.Linear(args.width, 10)]
            model = nn.Sequential(*layers)
        else:
            raise NotImplementedError

    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()

    return model, dataloaders, criterion