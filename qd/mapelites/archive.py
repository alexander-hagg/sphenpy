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

    def plot_phenotypes(self):


    def visualize_pyvista(phenotypes, domain, features=None, fitness=None, niches=None):
        nshapes = len(phenotypes)
        nrows = int(np.ceil(np.sqrt(nshapes)))
        shape=(nrows, nrows)
        # phenotypes, features = assign_niches(phenotypes, features, shape, domain)
        plotter = pv.Plotter(shape=shape, line_smoothing=True, polygon_smoothing=True)
        if features is not None:
            scaled_features = features
        #     print(f'features: {features}')
        #     for j in range(features.shape[1]):
        #         scaled_features[:,j] = maptorange.undo(features[:,j], domain['feat_ranges'][0][j], domain['feat_ranges'][1][j])
        for i in range(nshapes):
            if niches is None:
                row = int(np.floor(i / nrows))
                col = i % nrows
            else:
                row, col = niches
            if features is not None:
                feature_info = domain['labels'][0] + ': ' + str(scaled_features[i,0]) + 'm²\n' + \
                    domain['labels'][1] + ': ' + str(round(scaled_features[i,1], 2)) + \
                    'm² || preferred: ' + str(domain['target_area']) + 'm²\n'  # + \
                    # domain['labels'][2] + ': ' + str(scaled_features[i,2]) + \
                    # 'm² || preferred: low'
            else:
                feature_info = ""

            plotter.subplot(row, col)
            sz = domain['num_grid_cells']/2
            plotter.add_text(feature_info, font_size=12)

            if np.sum(phenotypes[i]) > 0:
                render_mesh(phenotypes[i])
                mesh = pv.read('mesh.stl')
                plotter.add_mesh(mesh)

            plane_mesh = pv.Plane(center=(sz,sz,0), direction=(0, 0, -1), i_size=2*sz, j_size=2*sz)
            sat = pv.read_texture('domain/nsg_cppn/mapsat.png')
            plotter.add_mesh(plane_mesh, texture=sat)
            # fitnesscolor = [1-1/(1+fitness[i][0]),1/(1+fitness[i][0]),0.0]
            fitnesscolor = [1-fitness[i][0],fitness[i][0],0.0]
            plotter.set_background(fitnesscolor, all_renderers=False)
        plotter.link_views()
        plotter.camera_position = [(50, 50, 10), (sz, sz, 0), (0, 0, 1)]
        plotter.show()


    def render_mesh(phenotype):
        model = VoxelModel(phenotype)  # , generateMaterials(4)  4 is aluminium.
        mesh = Mesh.fromVoxelModel(model)
        mesh.export('mesh.stl')