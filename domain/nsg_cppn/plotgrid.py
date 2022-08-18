from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def plot(phenotypes, domain, features=None, fitness=None, niches=None, filename=None):
    nshapes = len(phenotypes)
    nrows = int(np.ceil(np.sqrt(nshapes)))
    shape=(nrows, nrows)
    # phenotypes, features = assign_niches(phenotypes, features, shape, domain)
    plotter = pv.Plotter(off_screen=True, window_size=[8192, 6144], shape=shape, line_smoothing=True, polygon_smoothing=True)
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
        # print(f'fitness: {fitness[0]}')
        # quit()
        fitnesscolor = [1-fitness[0][i],fitness[0][i],0.0]
        plotter.set_background(fitnesscolor, all_renderers=False)
    plotter.link_views()
    plotter.camera_position = [(50, 50, 20), (sz, sz, 0), (0, 0, 1)]
    # if filename is not None:
        # plotter.save_graphic(filename, title='Solutions', raster=False, painter=False)
    plotter.show(screenshot=filename)
    # plt.imshow(plotter.image)
    # plt.show()
    return plotter


def render_mesh(phenotype):
    model = VoxelModel(phenotype)  # , generateMaterials(4)  4 is aluminium.
    mesh = Mesh.fromVoxelModel(model)
    mesh.export('mesh.stl')
