import numpy as np
from domain.nsg_cppn import express
EPSILON = 1e-5


def get(population, domain):
    # Express shapes
    phenotypes = express.do(population, domain)
    fitness = []
    features = []
    for i in range(len(phenotypes)):
        meter_squared_per_cell = (domain['substrate_length']/domain['num_grid_cells'])**2.0
        living_space_area = np.sum(phenotypes[i]) * meter_squared_per_cell
        # print(phenotypes[i])
        # print(f'meter_squared_per_cell: {meter_squared_per_cell} m²')
        # c = domain['num_grid_cells']
        # print(f'Maximum # cells per layer: {c**2.0}')
        # print(f'Maximum # cells: {3*c**2.0}')
        # print(f'# cells filled: {np.sum(phenotypes[i])}')
        # print(f'living_space_area: {living_space_area} m²')
        # quit()
        surface_area = np.sum(phenotypes[i], axis=2)
        # print(f'surface_area: {surface_area}')
        surface_area = surface_area > 0
        # print(f'surface_area: {surface_area}')
        surface_area = np.sum(surface_area)*meter_squared_per_cell
        # quit()
        windblock_area = np.sum(phenotypes[i], axis=0)
        windblock_area = windblock_area > 0
        windblock_area = np.sum(windblock_area)*meter_squared_per_cell
        if windblock_area == 0:
            windblock_area = 9999
        features.append([surface_area, living_space_area, windblock_area])

    features = np.asarray(features)
    fitness = 1/(1+features[:,2])
    fitness = np.transpose([fitness])
    # features = features[:,[domain['features'][0], domain['features'][1]]]
    # return fitness, features, phenotypes
    return fitness, features, phenotypes
