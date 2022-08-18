import numpy as np

def get(population, domain):
    features = np.zeros(shape=[len(population),2])
    fitness = np.zeros(shape=[len(population),1])
    phenotypes = []

    for i in range(len(population)):
        phenotypes.append(population[i].express())
        x = phenotypes[i][0]
        y = phenotypes[i][1]
        fitness[i] = 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2* np.pi * y)
        features[i,:] = [x,y]

    features = np.asarray(features)
    fitness = np.transpose(fitness)

    return fitness, features, phenotypes
