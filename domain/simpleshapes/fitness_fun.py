import numpy as np

def fitness_fun(population, domain):
    # Express shapes
    phenotypes = express(population, domain)
    # features = population
    # features[features > 1] = 1
    # features[features < 0] = 0
    # ranges = np.array(domain['par_ranges'])
    # x = population[:,0] * (ranges[1,0]-ranges[0,0]) + ranges[0,0]
    # y = population[:,1] * (ranges[1,1]-ranges[0,1]) + ranges[0,1]
    # fitness = 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2* np.pi * y)

    # fitrange = np.array(domain['fit_range'])
    # fitness = (fitness - fitrange[0])/(fitrange[1]-fitrange[0])
    # features = np.transpose(features)
    return fitness, features
