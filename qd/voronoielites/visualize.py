import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import numpy as np


def plot(archive, domain):
    print(archive['features'][:,1])
    f = interp2d(archive['features'][:,0],archive['features'][:,1],archive['fitness'],kind="cubic")
    # x_coords = np.arange(min(archive['features'][:,0]),max(archive['features'][:,0])+1)
    # z_coords = np.arange(min(archive['features'][:,1]),max(archive['features'][:,1])+1)
    # x_coords = np.arange(domain['feat_ranges'][0][0], domain['feat_ranges'][1][0])
    # z_coords = np.arange(domain['feat_ranges'][0][1], domain['feat_ranges'][1][1])
    x_coords = np.arange(0, 1)
    z_coords = np.arange(0, 1)
    print(x_coords)
    c_i = f(x_coords,z_coords)
    # plt.imshow(archive['fitness'], cmap='winter')
    fig = plt.imshow(c_i, extent=[0, 1, 0, 1], origin="lower", interpolation='bicubic')
    plt.scatter(archive['features'][:,0],archive['features'][:,1])
    # plt.clim(0,1)
    fig.axes.set_autoscale_on(True)
    plt.colorbar()
    plt.show()
