import numpy as np
from util import maptorange


def get(population, domain):
    features = np.copy(population)
    x = population[:,0]
    y = population[:,1]
    fitness = 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2* np.pi * y)

    features = np.asarray(features)
    for i in range(len(domain['feat_ranges'][0])):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])
    fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = np.transpose([fitness])

    return fitness, features
