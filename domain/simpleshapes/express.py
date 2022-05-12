# from domain.simpleshapes.get_natural_cubic_spline_model import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def do(genomes, domain):
    npoints = 51
    phenotypes = []
    for i in range(genomes.shape[0]):
        middle = int(len(genomes[i])/2)
        rho, phi = cart2pol(domain['base'][0], domain['base'][1])
        rho = rho * genomes[i,:middle]
        phi = phi + genomes[i,middle:]
        x, y = pol2cart(rho, phi)
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        tck, u = interpolate.splprep([x, y], s=0, k=3, per=True)
        x, y = interpolate.splev(np.linspace(0, 1, npoints), tck)
        phenotypes.append(np.asarray([x, y]))
    return phenotypes

def visualize(phenotype):
    plt.scatter(phenotype[0], phenotype[1])
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