import numpy as np


def niche_compete(fitness, features, archive, domain, config):
    # Discretize features into bins
    edges = np.linspace(domain['feat_ranges'][0], domain['feat_ranges'][1], num=config['resolution']-1)
    bin_assignment = np.empty((0,fitness.shape[1]), int)

    for i in range(edges.shape[1]):
        these_bins = np.digitize(features[:,i],edges[:,i])
        bin_assignment = np.vstack((bin_assignment, these_bins))

    # Find highest fitness per bin
    # Sort bins by fitness, then by bin coordinates
    bin_fitness = np.vstack([bin_assignment, fitness])
    num_features = bin_assignment.shape[0]
    idx = (-1*bin_fitness[num_features, :]).argsort()
    bin_fitness = bin_fitness[:, idx]
    for f in range(num_features - 1, -1, -1):
        idy = bin_fitness[f, :].argsort(kind='mergesort')
        bin_fitness = bin_fitness[:, idy]
        idx = idx[idy]
    unq, ind = np.unique(bin_fitness[0:2,:], return_inverse=False, return_index=True, axis=1)
    best_index = idx[ind]
    best_bin = bin_assignment[:, best_index]

    # Get replacement IDs in both archive and candidate arrays
    replaced = []
    replacement = []
    archive_fitness = archive.fitness
    for f in range(len(best_index)):
        bin_fitness = archive_fitness[best_bin[0, f],best_bin[1, f]]
        if np.isnan(bin_fitness) or bin_fitness < fitness[best_index[f]]:
            replacement.append(best_index[f])
            replaced.append([best_bin[0, f],best_bin[1, f]])

    return replaced, replacement
