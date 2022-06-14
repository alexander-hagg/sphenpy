import numpy as np
from domain.simpleshapes import express
from util import maptorange
import shapely.validation


def get(population, domain):
    # Express shapes
    phenotypes = express.do(population, domain)
    fitness = []
    features = []
    for i in range(len(phenotypes)):
        area = phenotypes[i].area
        perimeter = phenotypes[i].length
        symmetry = get_mirrorsymmetry(phenotypes[i])
        features.append([area, perimeter])
        fitness.append(symmetry)

    features = np.asarray(features)
    for i in range(len(domain['feat_ranges'][0])):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])
    fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = np.transpose([fitness])
    return fitness, features

def get_mirrorsymmetry(polygon):
    polygon_mirrored = shapely.affinity.scale(polygon, xfact=1.0, yfact=-1.0, origin='centroid')
    polygon_mirrored = shapely.validation.make_valid(polygon_mirrored)
    inters = polygon.intersection(polygon_mirrored)
    return inters.area/polygon.area