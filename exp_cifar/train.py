import argparse
from torch._C import _debug_set_autodiff_subgraph_inlining

from torch.utils.data import dataloader
from tasks import get_task, add_difficult_examples, extract_small_loader
import time
import os
import pandas as pd
import torch.optim as optim
import torch
import numpy as np
import copy
from utils import RunningAverageEstimator, get_binned_dataloaders
import math

from alignment import alignment

from nngeometry.object import PVector, PushForwardImplicit, PushForwardDense
from nngeometry.generator import Jacobian

import sys
sys.path.append('..')
from linearization_utils import LinearizationProbe

from nngeometry.object.vector import random_pvector

start_time = time.time()

parser = argparse.ArgumentParser(description='Compute various NTK alignment quantities')

parser.add_argument('--task', required=True, type=str, help='Task',
                    choices=['mnist_fc', 'cifar10_vgg19', 'cifar10_resnet18'])
parser.add_argument('--depth', default=0, type=int, help='network depth (only works with MNIST MLP)')
parser.add_argument('--width', default=0, type=int, help='network width (MLP) or base for channels (VGG)')

parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--alpha', default=1., type=float, help='Alpha = 1 -> Feature learning regime, Alpha = infty -> Lazy training')
parser.add_argument('--mom', default=0.9, type=float, help='Momentum')
parser.add_argument('--l2', default=0., type=float, help='Weight decay')

parser.add_argument('--diff', default=0., type=float, help='Proportion of difficult examples')
parser.add_argument('--diff-type', default='random', type=str, help='Type of difficult examples',
                    choices=['random', 'other'])

parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--batch_size', default=125, type=int, help='Batch size')
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--track_accs', action='store_true', help='Computes accuracy in subsets of trainset binned by cscore')
parser.add_argument('--track_aligns', action='store_true', help='Computes alignment in subsets of trainset binned by cscore')
parser.add_argument('--track_lin', action='store_true', help='Computes linearization measures')

parser.add_argument('--fork_from', default=None, type=str, help='Reload checkpoint')
parser.add_argument('--base_path', default=None, type=str, help='Path to store results')

args = parser.parse_args()

rae = RunningAverageEstimator()

def output_fn(x, t):
    return model(x)

def stopping_criterion(log):
    if (log.loc[len(log) - 1]['train_loss'] < 1e-2
            and log.loc[len(log) - 2]['train_loss'] < 1e-2):
        return True
    return False

def do_log(iterations, dataloaders, args):
    exps = np.arange(0, 300, .5)

    train_s = len(dataloaders['train'].dataset)
    mb_in_epoch = train_s // args.batch_size
    epoch = iterations // mb_in_epoch
    batch_idx = iterations % mb_in_epoch

    return (iterations == 0 or
            iterations in 5 * (1.15 ** exps).astype('int') or
            (epoch == (args.epochs-1) and (batch_idx == mb_in_epoch-1)) or
            batch_idx == 0)

def save_checkpoint(iterations, acc_milestone, model, model_0, optimizer, dataloaders, rae, alpha):
    checkpoint = {
        'model': model,
        'optimizer': optimizer,
        'rae': rae,
        'iterations': iterations,
        'targets': dataloaders['train'].dataset.tensors[1],
        'train_logits': get_logits(dataloaders['train_deterministic'], model, model_0, alpha),
        'test_logits': get_logits(dataloaders['test'], model, model_0, alpha),
        'torch_rng_state': torch.get_rng_state()
    }
    if args.diff > 0:
        checkpoint['train_diff_dataloader'] = dataloaders['mini_train_diff']
        checkpoint['train_diff_logits'] = get_logits(dataloaders['mini_train_diff'], model, model_0, alpha)
        checkpoint['train_easy_dataloader'] = dataloaders['mini_train_easy']
        checkpoint['train_easy_logits'] = get_logits(dataloaders['mini_train_easy'], model, model_0, alpha)
    checkpoint_path = os.path.join(results_dir, f'checkpoint_{acc_milestone}_{iterations}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f'saved checkpoint {checkpoint_path}')

def get_logits(dataloader, model, model_0, alpha):
    with torch.no_grad():
        total_logits = []
        for inputs, _, logits in dataloader:
            total_logits.append(alpha * (model(inputs) - model_0(inputs)) + logits)
        return torch.cat(total_logits)

# Training
def train(args, log, lin_log, model, model_0, optimizer, alpha, rae, start_iteration=0, next_milestone=10, lin_probe=None):
    model.train()
    model_0.train() # even if we are not actually training, needs to be put in train mode in order to kill batch norm fluctuations
    iterations = start_iteration
    # triggers checkpoint save after accuracy reaches milestone value
    do_save_checkpoint = False
    if iterations == 0:
        save_checkpoint(iterations, next_milestone, model, model_0, optimizer, dataloaders, rae, alpha)

    for epoch in range(args.epochs):

        print('\nEpoch: %d' % epoch)
        for batch_idx, (inputs, targets, logits) in enumerate(dataloaders['train']):

            if rae.get('train_acc') >= next_milestone / 100:
                do_save_checkpoint = True
                next_milestone += 5

            do_log_ = do_log(iterations, dataloaders, args)

            if do_log_:
                to_log = pd.Series()
                to_log['time'] = time.time() - start_time

                to_log['iteration'] = iterations
                to_log['epoch'] = epoch
                to_log['train_acc'], to_log['train_loss'] = rae.get('train_acc'), rae.get('train_loss')
                to_log['test_acc'], to_log['test_loss'] = \
                    test(dataloaders['mini_test'], model, model_0, alpha)
                if args.diff > 0:
                    to_log['train_diff_acc'], to_log['train_diff_loss'] = \
                        test(dataloaders['mini_train_diff'], model, model_0, alpha)
                    to_log['train_easy_acc'], to_log['train_easy_loss'] = \
                        test(dataloaders['mini_train_easy'], model, model_0, alpha)
                if args.track_accs:
                    to_log['train_accs'], to_log['train_losses'] = test_binned(dataloaders['train_binned'], model, model_0, alpha)
                    to_log['test_accs'], to_log['test_losses'] = test_binned(dataloaders['test_binned'], model, model_0, alpha)
                if args.track_aligns:
                    to_log['train_aligns'] = align_binned(dataloaders['train_binned'], model)
                log.loc[len(log)] = to_log
                print(log.loc[len(log) - 1])

                path_log = os.path.join(results_dir,'log.pkl')
                log.to_pickle(path_log)
                print(f'saved log to {path_log}')

                params_before = PVector.from_model(model).clone().detach()

            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            with torch.no_grad():
                outputs -= model_0(inputs).detach()
            outputs_alpha = alpha * outputs

            loss = criterion(outputs_alpha + logits, targets)
            grad_f = torch.autograd.grad(loss, outputs_alpha)[0]
            outputs.backward(grad_f)

            _, pred = (outputs_alpha + logits).max(1)
            acc = pred.eq(targets.view_as(pred)).float().mean()
            rae.update('train_loss', loss.item())
            rae.update('train_acc', acc.item())

            optimizer.step()
            iterations += 1

            if False and do_log_:
                params_after = PVector.from_model(model).clone().detach()
                to_log = pd.Series()
                to_log['iteration'] = iterations

                to_log['train_lin'], to_log['train_nonlin'] = analyze_lin(model, dataloaders['mini_train'],
                                                                          params_before, params_after)

                lin_log.loc[len(lin_log)] = to_log
                lin_log.to_pickle(os.path.join(results_dir,'lin_log.pkl'))
        if do_save_checkpoint:
            save_checkpoint(iterations, next_milestone-5, model, model_0, optimizer, dataloaders, rae, alpha)
            do_save_checkpoint = False


def test_binned(loaders, model, model_0, alpha):
    accs = []
    losses = []
    for loader in loaders:
        acc, loss = test(loader, model, model_0, alpha)
        accs.append(acc)
        losses.append(loss)
    return accs, losses


def align_binned(loaders, model):
    aligns = []
    for loader in loaders:
        aligns.append(alignment(model, lambda *x: model(x[0]), loader, 10, centering=False)[0])
    return aligns


def test(loader, model, model_0, alpha):
    model.eval()
    model_0.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, logits) in enumerate(loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = predict_logits(model, model_0, alpha, inputs, logits)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    model_0.train()
    return correct / total, test_loss / (batch_idx + 1)


def predict_logits(model, model_0, alpha, inputs, logits):
    with torch.no_grad():
        model0_output = model_0(inputs)
    return alpha * (model(inputs) - model0_output) + logits


def analyze_lin(model, dataloader, p_before, p_after):
    p_before.copy_to_model(model)
    with torch.no_grad():
        logits_before = []
        for inputs, _, _ in dataloader:
            logits_before.append(model(inputs))
        logits_before = torch.cat(logits_before)

    generator = Jacobian(model=model, n_output=10)
    pf = PushForwardImplicit(generator, examples=dataloader)
    d_logits = pf.mv((p_after - p_before).clone().detach())
    # dw = random_pvector(generator.layer_collection, device='cuda')
    # d_logits = pf.mv(dw)

    p_after.copy_to_model(model)
    with torch.no_grad():
        logits_after = []
        for inputs, _, _ in dataloader:
            logits_after.append(model(inputs))
        logits_after = torch.cat(logits_after)

    return d_logits.get_flat_representation().cpu(), (logits_after - logits_before).cpu()


def mkdir(p, m=None):
    try:
        os.mkdir(p)
    except:
        if m is not None:
            print(m)
        pass

name = ''
excludes = ['base_path']
for k, v in sorted(args.__dict__.items(), key=lambda a: a[0]):
    if k in excludes:
        continue
    if v is not False:
        if k == 'fork_from':
            if v is not None:
                name += 'fork=True,'
        else:
            name += '%s=%s,' % (k, str(v))
name = name[:-1]

if args.track_lin and args.fork_from is None:
    raise RuntimeError('I need a checkpoint to measure linearization quantities')

if args.fork_from is not None:
    checkpoint = torch.load(args.fork_from)

    parent_d, parent_f = os.path.split(args.fork_from)
    parent_checkpoint = parent_f.split('.')[0]
    mkdir(os.path.join(parent_d, 'children'))
    mkdir(os.path.join(parent_d, 'children', parent_checkpoint))
    next_milestone = int(parent_checkpoint.split('_')[1]) + 5

    results_dir = os.path.join(parent_d, 'children', parent_checkpoint, name)
    
    rae = checkpoint['rae']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    for g in optimizer.param_groups:
        g['lr'] = args.lr / args.alpha
        g['weight_decay'] = args.l2
        g['momentum'] = args.mom
    target_tensor = checkpoint['targets']
    train_logits = checkpoint['train_logits']
    test_logits = checkpoint['test_logits']

    start_iteration = checkpoint['iterations']

    _, dataloaders, criterion = get_task(args)

    train_dataset = dataloaders['train'].dataset
    train_dataset.tensors = (train_dataset.tensors[0], target_tensor, train_logits)
    test_dataset = dataloaders['test'].dataset
    test_dataset.tensors = (test_dataset.tensors[0], test_dataset.tensors[1], test_logits)
    if 'train_diff_logits' in checkpoint:
        train_diff_logits = checkpoint['train_diff_logits']
        dataloaders['mini_train_diff'] = checkpoint['train_diff_dataloader']
        train_diff_dataset = dataloaders['mini_train_diff'].dataset
        train_diff_dataset.tensors = (train_diff_dataset.tensors[0], train_diff_dataset.tensors[1], train_diff_logits)
        train_easy_logits = checkpoint['train_easy_logits']
        dataloaders['mini_train_easy'] = checkpoint['train_easy_dataloader']
        train_easy_dataset = dataloaders['mini_train_easy'].dataset
        train_easy_dataset.tensors = (train_easy_dataset.tensors[0], train_easy_dataset.tensors[1], train_easy_logits)

    torch.set_rng_state(checkpoint['torch_rng_state'])

    if args.track_lin:
        dataloaders['micro_test'] = extract_small_loader(dataloaders['test'], 100, 100)
        linprobe = LinearizationProbe(model, dataloaders['micro_test'])
        linprobe.buffer['signs_0'] = linprobe.get_signs().detach()
        linprobe.buffer['ntk_0'] = linprobe.get_ntk().detach()
        linprobe.buffer['repr_0'] = linprobe.get_last_layer_representation().detach()

else:
    checkpoint = None
    start_iteration = 0

    model, dataloaders, criterion = get_task(args)
    if args.diff > 0:
        add_difficult_examples(dataloaders, args)

    optimizer = optim.SGD(model.parameters(), lr=args.lr / args.alpha, momentum=args.mom,
                        weight_decay=args.l2)

    results_dir = os.path.join(args.base_path, name)

    rae.update('train_loss', -math.log(1/10))
    rae.update('train_acc', 1/10)

    next_milestone = 10

mkdir(results_dir, 'I will be overwriting a previous experiment')

dataloaders['mini_test'] = extract_small_loader(dataloaders['test'], 1000, 1000)
dataloaders['mini_train'] = extract_small_loader(dataloaders['train_deterministic'], 1000, 1000)

model_0 = copy.deepcopy(model)

columns = ['iteration', 'time', 'epoch',
           'train_loss', 'train_acc',
           'test_loss', 'test_acc']
columns_lin = ['iteration', 'train_lin', 'train_nonlin']
if args.diff > 0.:
    columns += ['train_diff_acc', 'train_diff_loss']
    columns += ['train_easy_acc', 'train_easy_loss']
if args.track_accs or args.track_aligns:
    with open('cifar10-cscores-orig-order.npz', 'rb') as f:
        d = np.load(f)
        cscores = d['scores']
        labels = d['labels']

    percentiles = [(i*.1, (i+1)*.1) for i in range(10)]

    rng = np.random.default_rng(1337)
    dataloaders['train_binned'] = get_binned_dataloaders(cscores[:40000],
        dataloaders['train_deterministic'], percentiles, 1000, rng)
    dataloaders['test_binned'] = get_binned_dataloaders(cscores[40000:],
        dataloaders['test'], percentiles, 1000, rng)

    if args.track_accs:
        columns += ['train_accs', 'train_losses']
        columns += ['test_accs', 'test_losses']
    if args.track_aligns:
        columns += ['train_aligns']

log = pd.DataFrame(columns=columns)
lin_log = pd.DataFrame(columns=columns_lin)
train(args, log, lin_log, model, model_0, optimizer, args.alpha, rae,
      start_iteration=start_iteration, next_milestone=next_milestone)
