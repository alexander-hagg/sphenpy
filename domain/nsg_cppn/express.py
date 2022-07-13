import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import util.voxCPPN.tools as voxvis
import math

def do_surf(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = cppn_out(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes

def cppn_out(genome, domain):
    if (np.isnan(genome)).any():
        return None
    X = np.arange(0, domain['grid_length'], 1)
    Y = np.arange(0, domain['grid_length'], 1)
    X, Y = np.meshgrid(X, Y)

    # TODO This should be replaced with a true CPPN (genome)
    Z = 3*np.sin(np.sqrt((X/2)**2 + (Y/2)**2))
    # Z[:] = 2
    # Z = 1*np.mod(X,4)+2
    phenotype = [X,Y,Z]
    print(X.shape)

    return phenotype


def do(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def express_single(genome, domain):
    X, Y, Z = cppn_out(genome, domain)
    # Convert to voxels
    print(f'Z: {Z}')
    voxels = np.zeros([domain['grid_length'], domain['grid_length'], domain['grid_length']])
    for x in range(domain['grid_length']):
        for y in range(domain['grid_length']):
            height = np.rint(Z[x,y])
            for z in range(int(height)):
                print(f'{x}/{y}/{z}/')
                voxels[x, y, math.floor(z)] = 1
    # voxels = np.rint(outputs).reshape(args.size, args.size, args.size)

    return voxels


def visualize_surf(phenotype, domain):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y, Z = phenotype
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()


def visualize(phenotype, domain):
    phenotype = phenotype.astype('int')
    phenotype = np.transpose(phenotype, axes=[1, 2, 0])
    phenotype = np.flip(phenotype, axis=1)
    print(f'voxels: {phenotype}')
    print(f'voxels.shape: {phenotype.shape}')

    np.save('tmp', phenotype)
    voxvis.render_voxels(np.pad(phenotype, 1, mode='empty'))
