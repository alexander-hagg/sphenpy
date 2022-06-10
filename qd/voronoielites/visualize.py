import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def plot(archive, domain):
    # grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    # grid_z2 = griddata(archive['features'], archive['fitness'], (grid_x, grid_y), method='cubic')
    # fig = plt.imshow(np.squeeze(grid_z2).T, extent=[0, 1, 0, 1], origin="lower", interpolation='bicubic')
    plt.scatter(archive['features'][:,0],archive['features'][:,1], c = archive['fitness'])
    plt.clim(0,1)
    # fig.axes.set_autoscale_on(False)
    cbar = plt.colorbar()
    cbar.set_label(domain['labels'][-1])
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    # plt.show()
    return plt
