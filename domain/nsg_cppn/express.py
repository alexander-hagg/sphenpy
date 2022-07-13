import matplotlib.pyplot as plt
import numpy as np
import util.voxCPPN.tools as voxvis


def do(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def express_single(genome, domain):
    if (np.isnan(genome)).any():
        return None
    x, y, z = np.indices((domain['plot_size'], domain['plot_size'], 4))
    for i in range(int(domain['num_units'])):
        pos_x = genome[i*domain['dof_perblock']]
        pos_y = genome[i*domain['dof_perblock']+1]
        height = genome[i*domain['dof_perblock']+2]

        pheno_x = (x > (pos_x)) & (x < (pos_x + domain['square_size'][0] + 1))
        pheno_y = (y > (pos_y)) & (y < (pos_y + domain['square_size'][1] + 1))
        floorplan = pheno_x & pheno_y

        building_height = (z > 0) & (z < (height+1))
        phenotype = (floorplan & building_height)

        if i == 0:
            total_phenotype = phenotype
        else:
            total_phenotype = total_phenotype | phenotype

    return total_phenotype


def visualize(phenotype, domain):
    phenotype = phenotype.astype('int')
    phenotype = np.transpose(phenotype, axes=[1, 2, 0])
    phenotype = np.flip(phenotype, axis=1)
    # np.save('tmp', phenotype)
    voxvis.render_voxels(phenotype)
