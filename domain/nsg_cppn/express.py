import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import util.voxCPPN.tools as voxvis

def do_surf(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = express_single_surface(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes

def express_single_surface(genome, domain):
    if (np.isnan(genome)).any():
        return None
    X = np.arange(0, domain['grid_length'], 1)
    Y = np.arange(0, domain['grid_length'], 1)
    X, Y = np.meshgrid(X, Y)

    # TODO This should be replaced with a true CPPN (genome)
    Z = 3*np.sin(np.sqrt((X/10)**2 + (Y/10)**2))
    phenotype = [X,Y,Z]

    return phenotype


def do(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def express_single(genome, domain):
    if (np.isnan(genome)).any():
        return None
    X = np.arange(0, domain['grid_length'], 1)
    Y = np.arange(0, domain['grid_length'], 1)
    X, Y = np.meshgrid(X, Y)

    # TODO This should be replaced with a true CPPN (genome)
    Z = 3*np.sin(np.sqrt((X/10)**2 + (Y/10)**2))
    phenotype = [X,Y,Z]
    # Convert to voxels
    print(f'Z: {Z}')
    voxels = np.zeros([domain['grid_length'], domain['grid_length'], domain['grid_length']])
    for x in range(domain['grid_length']):
        for y in range(domain['grid_length']):
            height = np.rint(Z[x,y])
            print(height)
            for z in range(int(height)):
                voxels[x, y, int(height)-z] = 1
    # voxels = np.rint(outputs).reshape(args.size, args.size, args.size)
    print(f'voxels: {voxels}')
    
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
    np.save('tmp', phenotype)
    voxvis.render_voxels(phenotype)
