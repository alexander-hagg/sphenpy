import numpy as np
import copy
import matplotlib.pyplot as plt

from templates.archive import archive
from templates.genome import genome


class mapelites_archive(archive):
    def __init__(self, domain, config):
        self.domain = domain
        self.config = config
        self.edges = []
        self.res = []
        for i in range(len(self.domain['features'])):
            self.edges.append(np.linspace(0, 1, self.config['resolution']))
            self.res.append(self.config['resolution'])

        self.total_niches = self.config['resolution']**len(self.domain['features'])
        self.fitness = np.full(self.res, np.nan)
        self.genes = np.full(self.res, genome)
        self.features = np.full(self.res, np.nan)
        self.features = np.expand_dims(self.features, 2)
        self.features = np.tile(self.features, (1, 1, len(domain['features'])))

    def update(self, fitness, features, genes):
        # Discretize features into bins
        edges = np.linspace(self.domain['feat_ranges'][0], self.domain['feat_ranges'][1], num=self.config['resolution']-1)
        bin_assignment = np.empty((0,fitness.shape[1]), int)

        for i in range(len(self.domain['features'])):
            j = self.domain['features'][i]
            these_bins = np.digitize(features[:,j],edges[:,i])
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
        for f in range(len(best_index)):
            bin_fitness = self.fitness[best_bin[0, f],best_bin[1, f]]
            if np.isnan(bin_fitness) or bin_fitness < fitness[0][best_index[f]]:
                replacement.append(best_index[f])
                replaced.append([best_bin[0, f],best_bin[1, f]])

        # Replace and add to archive
        for f in range(len(replacement)):
            self.fitness[replaced[f][0],replaced[f][1]] = fitness[0][replacement[f]]
        for f in range(len(replacement)):
            self.features[replaced[f][0],replaced[f][1], :] = features[replacement[f], :]
        for f in range(len(replacement)):
            self.genes[replaced[f][0],replaced[f][1]] = genes[replacement[f]]

        improvement = 100*len(replaced)/self.total_niches
        return improvement

    def create_pool(self):
        pool = copy.deepcopy(self.genes)
        pool = pool.reshape((pool.shape[0]*pool.shape[1], 1))

        # Remove empty genomes
        empties = [False] * pool.shape[0]
        for i in range(pool.shape[0]):
            if not isinstance(pool[i][0], genome):
                empties[i] = True
        pool = np.delete(pool, np.where(empties), axis=0)
        pool = pool.flatten()
        return pool

    def create_children(self):
        # Randomly select parents and copy to children
        pool = self.create_pool()

        selection = np.random.randint(0, pool.shape[0], self.config['num_children'])
        children = np.take(pool, selection, axis=0)
        children = np.squeeze(children).tolist()

        # Mutate children
        for child in children:
            child.mutate(self.config['mut_probability'], self.config['mut_sigma'])

        return children

    def get_niches(self):
        nonans = np.invert(np.isnan(self.fitness))
        return np.column_stack(np.where(nonans))

    def plot(self, ucbplot=False):
        plt.clf()
        plt.imshow(self.fitness, cmap='plasma')
        # if not ucbplot:
            # plt.clim(0, 1)
        plt.xlabel(self.domain['labels'][self.domain['features'][0]])
        plt.ylabel(self.domain['labels'][self.domain['features'][1]])
        cbar = plt.colorbar()
        if not ucbplot:
            cbar.set_label(self.domain['labels'][-1])
        else:
            cbar.set_label('Upper confidence bound')
        plt.show()
        return plt
