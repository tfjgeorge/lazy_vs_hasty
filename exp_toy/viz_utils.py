import torch

cmap = 'RdYlGn'

def custom_imshow(axis, ys, center=False):
    resolution = int(float(len(ys))**.5)
    if center:
        minmax = torch.max(torch.abs(ys)).item()
        return axis.imshow(ys.cpu().reshape(resolution, resolution).t(),
                           interpolation=None, cmap=cmap, extent=(-1, 1, -1, 1), alpha=.5,
                           origin='lower', vmin=-minmax, vmax=minmax)
    else:
        return axis.imshow(ys.cpu().reshape(resolution, resolution).t(),
                           interpolation=None, cmap=cmap, extent=(-1, 1, -1, 1), alpha=.5,
                           origin='lower')
