# %% [markdown]
# # Invariant Risk Minimization
# 
# This is an attempt to reproduce the "Colored MNIST" experiments from the
# paper [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
# by Arjovsky, et. al.

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
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
import os

import copy

import pickle as pkl

def makedir_lazy(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %% [markdown]
# ## Prepare the colored MNIST dataset
# 
# We define three environments (two training, one test) by randomly splitting the MNIST dataset in thirds and transforming each example as follows:
# 1. Assign a binary label y to the image based on the digit: y = 0 for digits 0-4
# and y = 1 for digits 5-9.
# 2. Flip the label with 25% probability.
# 3. Color the image either red or green according to its (possibly flipped) label.
# 4. Flip the color with a probability e that depends on the environment: 20% in
# the first training environment, 10% in the second training environment, and
# 90% in the test environment.

# %%
def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr], axis=2)
  return arr


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target, flip_color = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target, flip_color

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 5 != 0:
          continue
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      flip_label = np.random.uniform() < 0.05
      if flip_label:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        flip_color = np.random.uniform() < 0.10
      elif idx < 40000:
        # 10% in the second training environment
        flip_color = np.random.uniform() < 0.10
      else:
        # 90% in the test environment
        flip_color = True #np.random.uniform() < 0.9

      if flip_color:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label, int(flip_color)))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label, int(flip_color)))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label, int(flip_color)))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    makedir_lazy(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


# %% [markdown]
# ### Plot the data

# %%
def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 8))
  columns = 6
  rows = 3
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label, flip_color = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title(f"Label: {str(label)}, flipped: {flip_color} ")  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot
  


# %% [markdown]
# Plotting the train set

# %%
datapath = '~/slurm_tmpdir/data'

train1_set = ColoredMNIST(root=datapath, env='train1')
plot_dataset_digits(train1_set)

# %% [markdown]
# Plotting the test set

# %%
test_set = ColoredMNIST(root=datapath, env='test')
plot_dataset_digits(test_set)

# %% [markdown]
# Notice how the correlation between color and label are reversed in the train and test set.

# %% [markdown]
# ## Define neural network
# 
# The paper uses an MLP but a Convnet works fine too.

# %%
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(2 * 28 * 28, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 1)

  def forward(self, x):
    x = x.view(-1, 3 * 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(2, 20, 3, 1)
    self.conv2 = nn.Conv2d(20, 20, 3, 1)
    self.conv3 = nn.Conv2d(20, 20, 3, 1)
    self.fc1 = nn.Linear(20, 20)
    self.fc2 = nn.Linear(20, 20)
    self.fc3 = nn.Linear(20, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = F.relu(self.conv3(x))
    x = F.avg_pool2d(x, 2, 2)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


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

def loss_acc_by_group(output, target, flip_color):
    loss_indivs = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    loss_flipped = (loss_indivs * flip_color).sum() / flip_color.sum() 
    loss_unflipped = (loss_indivs * (1 - flip_color)).sum() / (1 - flip_color).sum() 

    acc_indivs = (output > 0) == target
    acc_flipped = (acc_indivs * flip_color).sum() / flip_color.sum() 
    acc_unflipped = (acc_indivs * (1 - flip_color)).sum() / (1 - flip_color).sum() 

    return loss_indivs.mean(), loss_flipped, loss_unflipped, acc_indivs.float().mean(), acc_flipped, acc_unflipped


def erm_train(model, model_0, alpha, device, train_loader, test_loader, optimizer, epoch, recorder):
    model.train()
    do_stop = 3
    for batch_idx, (data, target, flip_color) in enumerate(train_loader):
        data, target, flip_color = data.to(device), target.to(device).float(), flip_color.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            model_0_output = model_0(data)
        output = alpha * (model(data) - model_0_output)
        loss, loss_flipped, loss_unflipped, acc, acc_flipped, acc_unflipped = \
            loss_acc_by_group(output, target, flip_color)
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
                x_test, y_test, flip_test = x_test.to(device), y_test.to(device).float(), flip_test.to(device)
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
        if torch.isnan(loss) or loss.cpu() < .15:
            do_stop -= 1
            if do_stop == 0:
                return True
        else:
            do_stop = 3
    return False


def train_and_test_erm(alpha, model, model_0, all_train_loader, test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=0.02 / alpha**2, momentum=.9)
    recorder = Recorder()

    for epoch in range(1, 200):
        do_stop = erm_train(model, model_0, alpha, device, all_train_loader, test_loader, optimizer, epoch, recorder)

        # train_loss, train_acc = test_model(model, model_0, alpha, device, all_train_loader, set_name='train set')
        # test_loss, test_acc = test_model(model, model_0, alpha, device, test_loader)

        if do_stop:
            break
    return recorder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
all_train_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root=datapath, env='all_train',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, 0.1307), (0.3081, 0.3081))
                ])),
    batch_size=100, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root=datapath, env='test', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307), (0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=True, **kwargs)
model = ConvNet().to(device)

# alphas = 10**np.arange(0, 3.5, 1)
alphas = [0.5, 1, 2, 4]
recorders = []

model_0 = copy.deepcopy(model)

for alpha in alphas:
    print(f'------ alpha = {alpha}')
    model_copy = copy.deepcopy(model)
    recorder = train_and_test_erm(alpha, model_copy, model_0, all_train_loader, test_loader, device)
    recorders.append(recorder)

pkl.dump(recorders, open('./recorders.pkl', 'wb'))

# %%
recorders = pkl.load(open('recorders.pkl', 'rb'))

# %%
plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    ax = plt.plot(r.get('loss_flipped'), label=f'alpha={alpha}')
    plt.plot(r.get('loss_unflipped'), color=ax[0].get_color())
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
    ax = plt.plot(r.get('acc_flipped'), label=f'alpha={alpha}')
    plt.plot(r.get('acc_unflipped'), color=ax[0].get_color())
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
# plt.xlim(5e-1, 3)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.show()

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

smoothen = smoothen_running_average

# %%

plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    plt.scatter(smoothen(r.get('loss_unflipped')), smoothen(r.get('loss_flipped')), label=f'alpha={alpha}', marker='x', s=1)
plt.ylabel('loss flipped (examples that provide true signal)')
plt.xlabel('loss unflipped (spurious examples)')
plt.legend()
# plt.ylim(1e-1, 3)
# plt.xlim(1e-1, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.savefig('figures/loss_spur_vs_actual.pdf')
plt.show()

# %%
plt.figure(figsize=(10, 10))

for r, alpha in zip(recorders, alphas):
    plt.scatter(smoothen(r.get('acc_unflipped')), smoothen(r.get('acc_flipped')), label=f'alpha={alpha}', marker='x', s=1)
plt.ylabel('acc flipped (examples that provide true signal)')
plt.xlabel('acc unflipped (spurious examples)')
plt.legend()
# plt.ylim(1e-1, 3)
# plt.xlim(1e-1, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.savefig('figures/acc_spur_vs_actual.pdf')
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

# %%
