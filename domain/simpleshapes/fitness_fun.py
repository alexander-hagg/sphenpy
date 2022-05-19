import numpy as np
from domain.simpleshapes import express
from util import maptorange
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
import shapely.validation
import matplotlib.pyplot as plt


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
    for i in range(domain['nfeatures']):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])    
    fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = np.transpose([fitness])
    return fitness, features

def get_perimeter(polygon):
    """ returns the length of a polygon's perimeter defined by an ndarray of (x,y) coordinates """
    if polygon.type == 'MultiPolygon' or polygon.type == 'GeometryCollection':
        print("MP or GC")
        print(polygon.length)
    else:
        print("P")
        print(polygon.exterior.length)
        print(polygon.length)
    # perimeter = np.sum([euclidean(x, y) for x, y in zip(points, points[1:])])
    return polygon.length

def get_mirrorsymmetry(polygon):
    polygon_mirrored = shapely.affinity.scale(polygon, xfact=1.0, yfact=-1.0, origin='centroid')
    polygon_mirrored = shapely.validation.make_valid(polygon_mirrored)
    inters = polygon.intersection(polygon_mirrored)
    return inters.area/polygon.area