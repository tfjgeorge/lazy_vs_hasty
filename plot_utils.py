import matplotlib.pyplot as plt

padding = .1
linewidth = 6.75133

plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 7

def create_figure(n_per_row, ratio):
    fig_width = linewidth / n_per_row - 2 * padding
    fig_height = fig_width / ratio + 2 * padding
    return plt.figure(figsize=(fig_width, fig_height))

def save_fig(figure, path):
    figure.savefig(path, bbox_inches='tight', padding=padding)
