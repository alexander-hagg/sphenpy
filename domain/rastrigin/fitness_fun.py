import numpy as np
from util import maptorange


def get(population, domain):
    features = population
    # features[features > 1] = 1
    # features[features < 0] = 0
    for i in range(domain['nfeatures']):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])    
    ranges = np.array(domain['par_ranges'])
    x = population[:,0] * (ranges[1,0]-ranges[0,0]) + ranges[0,0]
    y = population[:,1] * (ranges[1,1]-ranges[0,1]) + ranges[0,1]
    fitness = 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2* np.pi * y)

    fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = np.transpose([fitness])
    return fitness, features
