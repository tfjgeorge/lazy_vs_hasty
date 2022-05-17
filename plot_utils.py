import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import interp1d
import numpy as np

padding = .1
linewidth = 5.50107

plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth'] = .5
plt.rcParams['patch.linewidth'] = .5
plt.rcParams['axes.labelsize'] = 5
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

def create_figure(width, ratio):
    # width is in proportion of whole line_width
    fig_width = linewidth * width - 2 * padding
    fig_height = fig_width / ratio + 2 * padding
    return plt.figure(figsize=(fig_width, fig_height),
                    constrained_layout=True)

def save_fig(figure, path):
    figure.savefig(path, bbox_inches='tight', padding=padding, dpi=1000)

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

    x_new = np.linspace(x.min(), x.max(), 200)


    x = np.maximum.accumulate(x)
    y = np.array(y)
    # x, y = rotate(x, y, np.pi/4, (.5, .5))
    y = sm.nonparametric.lowess(y, x, frac=.15, xvals=x_new)
    # x, y = rotate_back(x, y, np.pi/4, (.5, .5))

    # f = interp1d(x, y, bounds_error=False)

    # y_new = f(x_new)
    
    return x_new, y


def smoothen_interpolate(x, y):
    x_new = np.linspace(x.min(), x.max(), 200)
    f = interp1d(x, y, bounds_error=False)
    y_new = f(x_new)
    return x_new, y_new


def smoothen_moving_average(N):
    return lambda x: np.convolve(x, np.ones(N)/N, mode='valid')

def smoothen_running_average(gamma):
    def _f(x):
        o = []
        ra = x[0]
        for xi in x:
            ra = gamma * ra + (1 - gamma) * xi
            o.append(ra)
        return np.array(o)
    return _f

no_smoothing = lambda x: x
