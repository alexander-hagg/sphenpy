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
    x, y, z = np.indices((domain['plot_size'], domain['plot_size'], domain['plot_size']))
    for i in range(int(domain['dof']/2)):
        pos_x = genome[i*2]
        pos_y = genome[i*2+1]
        domain['square_size'][0]
        domain['square_size'][1]
        pheno_x = (x > (pos_x - 1)) & (x < (pos_x + domain['square_size'][0] + 1))
        pheno_y = (y > (pos_y - 1)) & (y < (pos_y + domain['square_size'][1] + 1))
        pheno_z = (z > 0) & (z < (1 + 1))
        phenotype = (pheno_x & pheno_y & pheno_z)
        
        if i == 0:
            total_phenotype = phenotype
        else:
            total_phenotype = total_phenotype | phenotype
    return total_phenotype

def visualize_raw(phenotype, color=[0.7, 0.7, 0.7], dx=0, dy=0):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(phenotype, facecolors=color, edgecolor='k')

    return plt

def visualize(phenotype, domain):
    visualize_raw(phenotype)
    # plt.axis('auto')
    # ax = plt.gca()
    # ax.set_xlim([0, domain['plot_size']])
    # ax.set_ylim([0, domain['plot_size']])
    # ax.set_zlim([0, 25])
    plt.show()
