from templates.genome import genome

import numpy as np
import random as rnd

from domain.nsg_cppn import cppn

import matplotlib.pyplot as plt
from matplotlib import cm
from util import maptorange

from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh

import pyvista as pv
EPSILON = 1e-5


class cppn_genome(genome):
    def __init__(self, domain):
        self.domain = domain
        self.genes = cppn.cppn(self.domain['num_neurons'], self.domain['num_layers'], self.domain['init_weight_variance'])

    def mutate(self, probability=0.1, sigma=1.0):
        with np.nditer(self.genes.activations, op_flags=['readwrite']) as it:
            for x in it:
                if rnd.random() < probability:
                    x[...] = rnd.randint(0,len(self.genes.act_funcs)-1)
        with np.nditer(self.genes.weights, op_flags=['readwrite']) as it:
            for x in it:
                if rnd.random() < probability:
                    x[...] = x[...] + rnd.gauss(0,sigma)

    def express(self):
        if self.genes is None:
            return None, None, None
        X = np.arange(0, self.domain['num_grid_cells'], 1)
        Y = np.arange(0, self.domain['num_grid_cells'], 1)
        X, Y = np.meshgrid(X, Y)
        raw_sample = self.genes.sample(self.domain['substrate'], self.domain)
        if self.domain['scale_cppn_out']:
            ranges = (np.max(raw_sample) - np.min(raw_sample))
            if ranges==0:
                ranges = 1
            Z = self.domain['max_height'] * (raw_sample - np.min(raw_sample)) / ranges
        else:
            Z = self.domain['max_height'] * raw_sample
            Z = np.floor(Z).astype(int)
            maximum = np.max(Z)
            Z = Z - (maximum - self.domain['max_height'])

        Z = Z.astype(int)

        # Convert to voxels
        voxels = np.zeros([self.domain['num_grid_cells'], self.domain['num_grid_cells'], self.domain['max_height']])
        if X is None:
            return voxels
        for x in range(self.domain['num_grid_cells']):
            for y in range(self.domain['num_grid_cells']):
                if self.domain['substrate'][x,y]:
                    for z in range(Z[x,y]):
                        voxels[x, y, z] = 1

        return voxels
