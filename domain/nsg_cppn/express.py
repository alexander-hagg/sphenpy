import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import util.voxCPPN.tools as voxvis
import math
from domain.nsg_cppn import cppn

EPSILON = 1e-5


def do_surf(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = cppn_out(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def cppn_out(net, domain):
    X = np.arange(0, domain['grid_length'], 1)
    Y = np.arange(0, domain['grid_length'], 1)
    X, Y = np.meshgrid(X, Y)
    binary_sample_grid = np.ones([domain['grid_length'],domain['grid_length']], dtype=bool)
    raw_sample = cppn.sample(binary_sample_grid, net, domain['grid_length'])
    # print(raw_sample)
    phenotype = [X,Y,raw_sample.astype(int)]
    if domain['scale_cppn_out']:
        ran = (np.max(raw_sample) - np.min(raw_sample))
        if ran==0:
            ran = EPSILON
        sample = domain['max_height'] * (raw_sample - np.min(raw_sample)) / ran
        phenotype = [X,Y,np.rint(sample).astype(int)]
    
    return phenotype


def do(genomes, domain):
    phenotypes = []
    for i in range(len(genomes)):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def express_single(genome, domain):
    X, Y, Z = cppn_out(genome, domain)
    # Convert to voxels
    voxels = np.zeros([domain['grid_length'], domain['grid_length'], domain['grid_length']])
    for x in range(domain['grid_length']):
        for y in range(domain['grid_length']):
            # height = np.rint(Z[x,y])
            for z in range(Z[x,y]):
                # voxels[x, y, math.floor(z)] = 1
                voxels[x, y, z] = 1

    return voxels


def visualize_surf(phenotype, domain):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y, Z = phenotype
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def visualize(phenotype, domain):
    phenotype = phenotype.astype('int')
    phenotype = np.transpose(phenotype, axes=[1, 2, 0])
    phenotype = np.flip(phenotype, axis=1)

    np.save('tmp', phenotype)
    voxvis.render_voxels(np.pad(phenotype, 1, mode='empty'))
