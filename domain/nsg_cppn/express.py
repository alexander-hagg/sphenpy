import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

import util.voxCPPN.tools as voxvis
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import generateMaterials

from domain.nsg_cppn import cppn
from domain.nsg_cppn import voxelvisualize

EPSILON = 1e-5


def do_surf(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = cppn_out(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def cppn_out(net, domain):
    X = np.arange(0, domain['grid_length'], 1)
    Y = np.arange(0, domain['grid_length'], 1)
    X, Y = np.meshgrid(X, Y)
    raw_sample = cppn.sample(domain['substrate'], net)
    if domain['scale_cppn_out']:
        ranges = (np.max(raw_sample) - np.min(raw_sample))
        if ranges==0:
            ranges = 1
        sample = domain['max_height'] * (raw_sample - np.min(raw_sample)) / ranges
        phenotype = [X,Y,sample.astype(int)]
    else:
        sample = domain['max_height'] * raw_sample
        sample = np.floor(sample).astype(int)
        maximum = np.max(sample)
        sample = sample - (maximum - domain['max_height'])
        phenotype = [X,Y,sample]

    return phenotype


def do(genomes, domain):
    phenotypes = []
    for i in range(len(genomes)):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes


def express_single(genome, domain):
    X, Y, Z = cppn_out(genome, domain)
    # Convert to voxels
    voxels = np.zeros([domain['grid_length'], domain['grid_length'], domain['max_height']])
    for x in range(domain['grid_length']):
        for y in range(domain['grid_length']):
            if domain['substrate'][x,y]:
                for z in range(Z[x,y]):
                    voxels[x, y, z] = 1

    return voxels


def visualize_surf(phenotype, domain):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y, Z = phenotype
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def visualize(phenotype, domain):
    phenotype = phenotype.astype('int')
    phenotype = np.transpose(phenotype, axes=[1, 2, 0])
    phenotype = np.flip(phenotype, axis=1)

    # np.save('tmp', phenotype)
    # voxvis.render_voxels(np.pad(phenotype, 1, mode='empty'))
    voxelvisualize.render_voxels(np.pad(phenotype, 1, mode='empty'))
    # render_mesh(phenotype)

def render_mesh(phenotype):
    # phenotype = np.pad(phenotype, 1, mode='empty')
    model = VoxelModel(phenotype)  #4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    mesh.export('mesh.stl')