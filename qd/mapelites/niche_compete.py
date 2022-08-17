import numpy as np

def niche_compete(fitness, features, archive, domain, config):
    # print(features)
    # assert (features >= 0).all(),"Feature values smaller than 0, assumed to be normalized between [0,1]"
    # assert (features <= 1).all(),"Feature values larger than 1, assumed to be normalized between [0,1]"
    # assert (fitness >= 0).all(),"Fitness values smaller than 0, assumed to be normalized between [0,1]"
    # assert (fitness <= 1).all(),"Fitness values larger than 1, assumed to be normalized between [0,1]"

    # Discretize features into bins
    # edges = np.linspace(0, 1, num=config['resolution'])
    feat_ranges = domain['feat_ranges']
    # print(feat_ranges)
    # print(domain['features'])
    feat_ranges = feat_ranges[0][domain['features']]
    # print(f'feat_ranges: {feat_ranges}')
    edges = np.linspace(domain['feat_ranges'][0], domain['feat_ranges'][1], num=config['resolution'])
    # print(f'edges: {edges[:,0]}')
    # print(f'features: {features[:,0]}')
    bin_assignment = []
    for i in range(edges.shape[1]):
        these_bins = np.digitize(features[:,i],edges[:,i])
        print(f'these_bins: {these_bins}')
        bin_assignment = np.hstack((bin_assignment, these_bins))
    bin_assignment = bin_assignment - 1
    
    # print(f'features: {features}')
    # print(f'bin_assignment: {bin_assignment}')
    # quit()
    
    ## Find highest fitness per bin
    # Sort bins by fitness, then by bin coordinates
    bin_fitness = np.hstack([bin_assignment, fitness])
    num_features = bin_assignment.shape[1]
    idx = (-1*bin_fitness[num_features, :]).argsort()
    bin_fitness = bin_fitness[:, idx]
    for f in range(num_features - 1, -1, -1):
        idy = bin_fitness[f, :].argsort(kind='mergesort')
        bin_fitness = bin_fitness[:, idy]
        idx = idx[idy]
    unq, ind = np.unique(bin_fitness[0:2,:], return_inverse=False, return_index=True, axis=1)
    best_index = idx[ind]
    best_bin = bin_assignment[best_index, :]

    # Get replacement IDs in both archive and candidate arrays
    replaced = []
    replacement = []
    for f in range(len(best_index)):
        archive_fitness = archive['fitness']
        bin_fitness = archive_fitness[best_bin[f, 0],best_bin[f, 1]]
        if np.isnan(bin_fitness) or bin_fitness < fitness[best_index[f]]:
            replacement.append(best_index[f])
            replaced.append([best_bin[f, 0],best_bin[f, 1]])

    return replaced, replacement
