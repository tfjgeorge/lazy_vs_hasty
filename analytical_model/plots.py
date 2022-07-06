# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from argparse import Namespace
from functools import reduce

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

# %%
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.cuda.set_device(0)

# %%
def generate_solution(t, array_sqrmu, array_y, sigma, linear = False):
  # compute components of the error
  if linear:
    err =  np.exp(-2*t*sigma*(array_sqrmu**2))*(sigma*array_sqrmu - array_y)
  else: 
    exp = np.exp(-2*t*array_sqrmu*array_y)
    err = exp*array_y*np.divide(sigma*array_sqrmu - array_y, sigma*array_sqrmu - exp*(sigma*array_sqrmu - array_y))

  return err**2 

## Example 1
n = 20
sigma = 0.001
array_sqrmu = np.sqrt(np.arange(1,n+1)[::-1])
array_y = np.arange(1,n+1)

tlin_list = np.linspace(0, 50, num=200)
err_lin = list(map(lambda t: generate_solution(t, array_sqrmu, array_y, sigma, linear = True), tlin_list))
err_lin = [[err_lin[k][j] for k in range(len(tlin_list))] for j in range(n)]

tnonlin_list = np.linspace(0, 0.5, num=200)
err_nonlin = list(map(lambda t: generate_solution(t, array_sqrmu, array_y, sigma), tnonlin_list))
err_nonlin = [[err_nonlin[k][j] for k in range(len(tnonlin_list))] for j in range(n)]

## Example 2


# %%
# Example 1
plt.figure()
plt.title('Linear')
for j in range(5):
  plt.plot(tlin_list, err_lin[j])
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
  #plt.plot(t_list, err_nonlin, label='non-linear')
#plt.xlabel("Training Iteration")
#plt.ylabel("Mode errors")
#plt.legend(['n='+str(j) for j in range(1, n+1)])
plt.show()

fig = plt.figure()
plt.title('Non Linear')
for j in range(5):
  #plt.plot(t_list, err_lin, label='linear')
  plt.plot(tnonlin_list, err_nonlin[j])
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
#plt.xlabel("Training Iteration")
#plt.ylabel("Mode errors")
#plt.legend(['n='+str(j) for j in range(1, n+1)])
plt.show()

# Example 2


# %%
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(type(self), self).__init__()
        
        self.input_size = input_size
        self.act = F.relu
        
        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, 1)
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = nn.Linear(hidden_sizes[-1], 1)
            

    def forward(self, x, return_feats=False, logits=False):
        feats = []
        for layer in self.hidden_layers:
            x = self.act(layer(x))
            feats.append(x)
        x = self.output_layer(x)
        if not logits:
            x = torch.sigmoid(x)
        if return_feats:
            return x.flatten(), feats
        return x.flatten()

# %%



