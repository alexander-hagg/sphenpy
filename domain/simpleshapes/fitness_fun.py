import numpy as np
from domain.simpleshapes import express
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
from shapely.validation import make_valid
import matplotlib.pyplot as plt

def get(population, domain):
    # Express shapes
    phenotypes = express.do(population, domain)
    for i in range(len(phenotypes)):
        area = get_area_of_polygon(phenotypes[i])
        print(f'Area: {area}')
        # Circumference
        perimeter = get_perimeter(phenotypes[i])
        print(f'perimeter: {perimeter}')
        # Mirror symmetry
        symmetry = get_mirrorsymmetry(phenotypes[i])
        print(f'symmetry: {symmetry}')
        # features[features > 1] = 1
        # features[features < 0] = 0
        # fitrange = np.array(domain['fit_range'])
        # fitness = (fitness - fitrange[0])/(fitrange[1]-fitrange[0])
        # features = np.transpose(features)
    return fitness, features


def get_area_of_polygon(points):
    """Calculates the area of an arbitrary polygon given its vertices"""
    # area = 0.0
    # for i in range(-1, len(points[0])-1):
    #     area += points[0][i] * (points[1][i+1] - points[1][i-1])
    # return abs(area) / 2.0
    points = np.transpose(points)
    p1 = Polygon(points)
    return p1.area


def get_perimeter(points):
    """ returns the length of a polygon's perimeter defined by an ndarray of (x,y) coordinates """
    perimeter = np.sum([euclidean(x, y) for x, y in zip(points, points[1:])])
    return perimeter

def get_mirrorsymmetry(points):
    points = np.transpose(points)
    p1 = Polygon(points)
    plt.plot(*p1.exterior.xy)
    #plt.show()
    points[:,1] = -points[:,1]
    p2 = Polygon(points)
    plt.plot(*p2.exterior.xy)
    # plt.show()
    p3 = p1.intersection(p2)
    plt.plot(*p3.exterior.xy)
    # plt.show()
    return p3.area/p1.area