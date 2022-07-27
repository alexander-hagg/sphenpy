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
        # print(phenotypes[i])
        volume = np.sum(phenotypes[i])
        print(f'volume: {volume}')
        volume_error = (volume-domain['target_volume'])
        print(f'volume_error: {volume_error}')

        surface_area = np.sum(phenotypes[i], axis=2)
        surface_area = surface_area > 0
        surface_area = np.sum(surface_area.astype(int))
        print(f'surface_area: {surface_area}')
        

        windblock_area = np.sum(phenotypes[i], axis=0)
        windblock_area = windblock_area > 0
        windblock_area = np.sum(windblock_area.astype(int))
        print(f'windblock_area: {windblock_area}')
        fit = 1/(1+windblock_area)
        print(f'Fitness: {fit}')
        features.append([surface_area, volume_error, windblock_area])
        fitness.append(fit)

    features = np.asarray(features)
    # for i in range(len(domain['feat_ranges'][0])):
    #     features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])
    # fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = np.transpose([fitness])
    return fitness, features, phenotypes

