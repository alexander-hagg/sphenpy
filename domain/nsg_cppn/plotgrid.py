from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
import pyvista as pv
import numpy as np


def plot(phenotypes, domain, features=None, fitness=None, niches=None, filename=None, gridresolution=None, output_resolution=[8192, 6144]):
    nshapes = len(phenotypes)
    if niches is not None:
        if gridresolution is not None:
            nrows = gridresolution
            ncols = gridresolution
        else:
            nrows = np.max(niches[:,0])+1
            ncols = np.max(niches[:,1])+1
    else:
        nrows = int(np.ceil(np.sqrt(nshapes)))
        ncols = nrows
    shape=(nrows, ncols)
    plotter = pv.Plotter(off_screen=True, window_size=output_resolution, shape=shape, line_smoothing=True, polygon_smoothing=True)
    for i in range(nshapes):
        if niches is None:
            row = int(np.floor(i / nrows))
            col = i % nrows
        else:
            row, col = niches[i]
        if features is not None:
            feature_info = domain['labels'][0] + ': ' + str(round(features[i,0])) + 'm²\n' + \
                domain['labels'][1] + ': ' + str(round(features[i,1], 2))
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
        # clrscale = 10*fitness[0][i]
        # if clrscale > 1.0:
        #     clrscale = 1.0
        clrscale = fitness[0][i]
        fitnesscolor = [1-clrscale,clrscale,0.0]
        plotter.set_background(fitnesscolor, all_renderers=False)
    plotter.link_views()
    plotter.camera_position = [(50, 50, 20), (sz, sz, 0), (0, 0, 1)]
    plotter.show(screenshot=filename)
    return plotter


def render_mesh(phenotype):
    model = VoxelModel(phenotype)  # , generateMaterials(4)  4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    mesh.export('mesh.stl')
