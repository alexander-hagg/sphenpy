# from domain.simpleshapes.get_natural_cubic_spline_model import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def do(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes

def express_single(genome, domain):
    if (np.isnan(genome)).any():
        return None
    npoints = 51
    middle = int(len(genome)/2)
    rho, phi = cart2pol(domain['base'][0], domain['base'][1])
    rho = rho * genome[:middle]
    phi = phi + genome[middle:]
    x, y = pol2cart(rho, phi)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    tck, u = interpolate.splprep([x, y], s=0, k=3, per=True)
    x, y = interpolate.splev(np.linspace(0, 1, npoints), tck)
    return np.asarray([x, y])

def visualize_raw(phenotype, color=[0, 0, 0]):
    plt.fill(phenotype[0], phenotype[1], color=color)
    
def visualize(phenotype):
    visualize_raw(phenotype)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.show()

def cart2pol(x, y):
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)