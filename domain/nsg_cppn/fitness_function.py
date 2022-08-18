import numpy as np
EPSILON = 1e-5


def get(list_genomes, domain):
    # Express shapes
    features = np.zeros(shape=[len(list_genomes),2])
    fitness = np.zeros(shape=[len(list_genomes),1])
    phenotypes = []

    for i in range(len(list_genomes)):
        phenotypes.append(list_genomes[i].express())
        meter_squared_per_cell = (domain['substrate_length']/domain['num_grid_cells'])**2.0
        living_space_area = np.sum(phenotypes[i]) * meter_squared_per_cell
        surface_area = np.sum(phenotypes[i], axis=2)
        surface_area = surface_area > 0
        surface_area = np.sum(surface_area)*meter_squared_per_cell
        windblock_area = np.sum(phenotypes[i], axis=0)
        windblock_area = windblock_area > 0
        windblock_area = np.sum(windblock_area)*meter_squared_per_cell
        if windblock_area == 0:
            windblock_area = 9999
        features[i,:] = ([surface_area, living_space_area])
        fitness[i] = 1/(1+windblock_area)

    fitness = np.transpose(fitness)
    return fitness, features, phenotypes
