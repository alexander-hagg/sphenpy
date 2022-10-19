import numpy as np
from scipy.ndimage import label
from util import maptorange

EPSILON = 1e-5


def get(list_genomes, domain):
    # Express shapes
    # features = np.zeros(shape=[len(list_genomes),2])
    fitness = np.zeros(shape=[len(list_genomes), 1])
    rawfeatures = np.zeros(shape=[len(list_genomes), 4])
    phenotypes = []

    for i in range(len(list_genomes)):
        phenotypes.append(list_genomes[i].express())
        meter_squared_per_cell = (
            domain["substrate_length"] / domain["num_grid_cells"]
        ) ** 2.0 * 4.5
        # print(phenotypes[i])
        # print(f'meter_squared_per_cell: {meter_squared_per_cell}')
        # print(f'np.sum(phenotypes[i]): {np.sum(phenotypes[i])}')
        living_space_area = np.sum(phenotypes[i]) * meter_squared_per_cell
        # print(f'living_space_area: {living_space_area}')
        surface_area = np.sum(phenotypes[i], axis=2)
        surface_area = surface_area > 0
        surface_area = np.sum(surface_area) * meter_squared_per_cell
        windblock_area = np.sum(phenotypes[i], axis=0)
        windblock_area = windblock_area > 0
        windblock_area = np.sum(windblock_area) * meter_squared_per_cell
        if windblock_area == 0:
            windblock_area = 9999

        connection_directions = np.ones((3, 3), dtype=np.int)
        labeled, num_structures = label(
            np.sum(phenotypes[i], axis=2), connection_directions
        )

        rawfeatures[i, :] = [
            surface_area,
            num_structures,
            living_space_area,
            windblock_area,
        ]
        fitness[i] = (
            0.5 / (1 + windblock_area)
            + 0.5 / (1 + np.abs(living_space_area - domain["target_area"]))
        ) ** (1 / 5)
        # fitness[i] = 1/(1+np.abs(living_space_area-domain['target_area']))**(1/5)

    features = rawfeatures[:, [domain["features"][0], domain["features"][1]]]
    # print(f'features: {features}')
    for fid in range(features.shape[1]):
        features[:, fid] = maptorange.do(
            features[:, fid],
            domain["feat_ranges"][0][domain["features"][fid]],
            domain["feat_ranges"][1][domain["features"][fid]],
        )
    # print(f'features: {features}')
    fitness = np.transpose(fitness)
    return fitness, features, phenotypes, rawfeatures
