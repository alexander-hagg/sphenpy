import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from shapely.geometry import Polygon
import shapely.validation
from descartes import PolygonPatch


def do(genomes, domain):
    phenotypes = []
    for i in range(genomes.shape[0]):
        phenotype = express_single(genomes[i], domain)
        phenotypes.append(phenotype)
    return phenotypes

def express_single(genome, domain):
    if (np.isnan(genome)).any():
        return None
    npoints = 51
    middle = int(len(genome)/2)
    rho, phi = cart2pol(domain['base'][0], domain['base'][1])
    rho = rho * genome[:middle]
    phi = phi + genome[middle:]
    x, y = pol2cart(rho, phi)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    tck, u = interpolate.splprep([x, y], s=0, k=3, per=True)
    x, y = interpolate.splev(np.linspace(0, 1, npoints), tck)
    points = np.transpose(np.asarray([x, y]))
    p1 = Polygon(points)
    p1 = shapely.validation.make_valid(p1)
    return p1

def visualize_raw(phenotype, color=[0, 0, 0], dx=0, dy=0):
    if phenotype.type == 'Polygon':
        x, y = zip(*list(phenotype.exterior.coords))
        plt.fill(x+dx, y+dy, color=color)
    if phenotype.type == 'MultiPolygon':
        for i in range(len(phenotype.geoms)):
            x, y = zip(*list(phenotype.geoms[i].exterior.coords))
            plt.fill(x+dx, y+dy, color=color)
    if phenotype.type == 'GeometryCollection':
        for i in range(len(phenotype.geoms)):
            if not phenotype.geoms[i].type == 'Point' and not phenotype.geoms[i].type == 'LineString':
                if phenotype.geoms[i].type == 'MultiPolygon':
                    for j in range(len(phenotype.geoms[i].geoms)):
                        x, y = zip(*list(phenotype.geoms[i].geoms[j].exterior.coords))
                        plt.fill(x+dx, y+dy, color=color)
                else:
                    x, y = zip(*list(phenotype.geoms[i].exterior.coords))
                    plt.fill(x+dx, y+dy, color=color)

    
def visualize(phenotype):
    visualize_raw(phenotype)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.show()

def cart2pol(x, y):
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)