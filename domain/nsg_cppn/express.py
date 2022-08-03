import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

# import util.voxCPPN.tools as voxvis
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import generateMaterials

import pyvista as pv
from pyvista import examples

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


def visualize(phenotype, domain, features=None):
    phenotype = phenotype.astype('int')
    phenotype = np.transpose(phenotype, axes=[1, 2, 0])
    phenotype = np.flip(phenotype, axis=1)

    # np.save('tmp', phenotype)
    # voxvis.render_voxels(np.pad(phenotype, 1, mode='empty'))
    if features is not None:
        feature_info = domain['labels'][0] + ': ' + str(features[0]) + 'm²<br />' + \
            domain['labels'][1] + ': ' + str(round(features[1], 2)) + \
            'm² || preferred: ' + str(domain['target_area']) + 'm²<br />' + \
            domain['labels'][2] + ': ' + str(features[2]) + \
            'm² || preferred: low'
        print(feature_info)
    else:
        feature_info = ""
    # voxelvisualize.render_voxels(np.pad(phenotype, 1, mode='empty'), feature_info)
    render_mesh(phenotype)
    # render_mesh_env(phenotype)
    # export_vtk(phenotype)


def visualize_pyvista(phenotypes, domain, features=None):
    nshapes = len(phenotypes)
    nrows = int(np.ceil(np.sqrt(nshapes)))
    plotter = pv.Plotter(shape=(nrows, nrows))
    for i in range(nshapes):
        row = int(np.floor(i / nrows))
        col = i % nrows
        if features is not None:
            feature_info = domain['labels'][0] + ': ' + str(features[i,0]) + 'm²\n' + \
                domain['labels'][1] + ': ' + str(round(features[i,1], 2)) + \
                'm² || preferred: ' + str(domain['target_area']) + 'm²\n' + \
                domain['labels'][2] + ': ' + str(features[i,2]) + \
                'm² || preferred: low'
        else:
            feature_info = ""

        plotter.subplot(row, col)
        if np.sum(phenotypes[i]) == 0:
            plotter.add_text('Invalid shape not displayed', font_size=8)
        else:
            plotter.add_text(feature_info, font_size=8)
            render_mesh(phenotypes[i])
            mesh = pv.read('mesh.stl')
            plotter.add_mesh(mesh)
            sz = domain['grid_length']/2
            sz = sz
            plane_mesh = pv.Plane(center=(sz,sz,0), direction=(0, 0, -1), i_size=2*sz, j_size=2*sz)
            sat = pv.read_texture('domain/nsg_cppn/mapsat.png')
            print(sat)
            plotter.add_mesh(plane_mesh, texture=sat)
            fitnesscolor = [1-1/(1+features[i,2]/10),1/(1+features[i,2]/10),0.0]
            plotter.set_background(fitnesscolor, all_renderers=False)
    plotter.link_views()
    plotter.camera_position = [(50, 50, 10), (sz, sz, 0), (0, 0, 1)]
    plotter.show()


def render_mesh(phenotype):
    model = VoxelModel(phenotype)  #, generateMaterials(4)  4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    mesh.export('mesh.stl')


def render_mesh_full(phenotype):
    # fullmodel
    # phenotype
    model = VoxelModel(fullmodel)  #, generateMaterials(4)  4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    mesh.export('mesh.stl')
