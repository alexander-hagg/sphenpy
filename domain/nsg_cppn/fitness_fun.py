import numpy as np
from domain.nsg_cppn import express
from util import maptorange
EPSILON = 1e-5


def get(population, domain):
    # Express shapes
    phenotypes = express.do(population, domain)
    fitness = []
    features = []
    for i in range(len(phenotypes)):
        living_space_area = np.sum(phenotypes[i])
        meter_per_cell = 93/domain['grid_length']
        living_space_area = living_space_area * meter_per_cell**2.0
        surface_area = np.sum(phenotypes[i], axis=2)
        surface_area = surface_area > 0
        surface_area = np.sum(surface_area.astype(int))
        windblock_area = np.sum(phenotypes[i], axis=0)
        windblock_area = windblock_area > 0
        windblock_area = np.sum(windblock_area.astype(int))
        features.append([surface_area, living_space_area, windblock_area])

    features = np.asarray(features)
    for i in range(len(domain['feat_ranges'][0])):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])

    # fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = 1/(1+features[:,2])
    fitness = np.transpose([fitness])
    # features = features[:,[domain['features'][0], domain['features'][1]]]
    # return fitness, features, phenotypes
    return fitness, features, phenotypes
