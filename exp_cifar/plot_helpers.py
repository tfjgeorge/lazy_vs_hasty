import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from plot_utils import *

import pandas as pd
import os

def concatenate_acc_loss(d, train=True):
    accs = []
    losses = []
    for i, r in d.iterrows():
        accs.append(r[f'{"train" if train else "test"}_accs'])
        losses.append(r[f'{"train" if train else "test"}_losses'])

    accs = np.array(accs)
    losses = np.array(losses)

    return accs, losses


def smoothen_2d_homemade(xs, ys, step=.03):
    x_o = [xs[0]]
    y_o = [ys[0]]
    length = 0
    next_step = step
    x_c, y_c = xs[1], ys[1]
    next_i = 2

    while True:
        x_p, y_p = x_o[-1], y_o[-1]
        length_this_segment = ((x_c - x_p)**2 + (y_c - y_p)**2)**.5
        if length_this_segment > step:
            step_interpolate = min(step / length_this_segment, 1)
            # interpolate between p and c
            x_o.append(x_p * (1 - step_interpolate) + x_c * step_interpolate)
            y_o.append(y_p * (1 - step_interpolate) + y_c * step_interpolate)
            length += step
            next_step += step
        else:
            # length += length_this_segment
            x_c, y_c = xs[next_i], ys[next_i]
            next_i += 1
            if next_i == len(xs):
                break
    x_o.append(x_c)
    y_o.append(y_c)
    # return xs, ys
    return x_o, y_o

smoothen_2d = smoothen_2d_homemade