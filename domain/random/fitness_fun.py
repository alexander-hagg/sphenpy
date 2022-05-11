import numpy as np

def fitness_fun(population, domain):
    fitness = np.random.rand(population.shape[0])
    fitness[fitness > 1] = 1
    fitness[fitness < 0] = 0
    features = np.random.rand(domain['nfeatures'],population.shape[0])
    features[features > 1] = 1
    features[features < 0] = 0
    return fitness, features
