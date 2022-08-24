import numpy as np
import copy
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

# For interpolation plotting
from scipy.interpolate import griddata

# For Voronoi plotting
from scipy.spatial import Voronoi, voronoi_plot_2d

from templates.archive import archive
from templates.genome import genome


class voronoi_archive(archive):
    def __init__(self, domain, config):
        self.domain = domain
        self.config = config
        self.total_niches = self.config['resolution']

        self.fitness = np.empty([0, 1])
        self.genes = np.empty([0], genome)
        self.features = np.empty([0, len(domain['features'])])

    def update(self, fitness, features, genes):
        all_genomes = np.hstack((self.genes,np.asarray(genes)))
        all_features = np.vstack((self.features,features))
        all_fitness = np.vstack((self.fitness,np.transpose(np.asarray(fitness))))
        while all_features.shape[0] > self.config['resolution']:
            distances = cdist(all_features, all_features)
            ind = np.unravel_index(np.argmin(distances[np.where(~np.eye(distances.shape[0],dtype=bool))], axis=None), distances.shape)
            lower = all_fitness[ind[0]] > all_fitness[ind[1]]
            all_genomes = np.delete(all_genomes, ind[int(lower)], 0)
            all_features = np.delete(all_features, ind[int(lower)], 0)
            all_fitness = np.delete(all_fitness, ind[int(lower)], 0)

        # Update archive
        self.genes = all_genomes
        self.features = all_features
        self.fitness = all_fitness

        improvement = None
        return improvement

    def create_pool(self):
        pool = copy.deepcopy(self.genes)

        # Remove empty genomes
        empties = [False] * pool.shape[0]
        for i in range(pool.shape[0]):
            if not isinstance(pool[i], genome):
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
        voronoi = Voronoi(self.features)
        voronoi_plot_2d(voronoi, show_vertices=False, show_points=False)
        # for region in voronoi.regions:
            # if -1 not in region:
                # polygon = [voronoi.vertices[p] for p in region]
                # plt.fill(*zip(*polygon))
        plt.scatter(self.features[:,0], self.features[:,1], c=self.fitness, zorder=100)
        # fig = voronoi_plot_2d(vor, show_vertices=False)
        # plt.xlabel(self.domain['labels'][self.domain['features'][0]])
        # plt.ylabel(self.domain['labels'][self.domain['features'][1]])
        # cbar = plt.colorbar()
        # if not ucbplot:
        #     cbar.set_label(self.domain['labels'][-1])
        # else:
        #     cbar.set_label('Upper confidence bound')
        plt.show()
        return plt

    def plot_interp(self, ucbplot=False):
        X, Y = np.mgrid[0:1:100j, 0:1:100j]        
        grid_z2 = griddata(self.features, self.fitness, (X, Y), method='cubic')
        plt.imshow(grid_z2.T[0], extent=(0, 1, 0, 1), origin='lower')
        plt.xlabel(self.domain['labels'][self.domain['features'][0]])
        plt.ylabel(self.domain['labels'][self.domain['features'][1]])
        cbar = plt.colorbar()
        if not ucbplot:
            cbar.set_label(self.domain['labels'][-1])
        else:
            cbar.set_label('Upper confidence bound')
        plt.show()
        return plt
