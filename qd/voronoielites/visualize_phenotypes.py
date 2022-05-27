import matplotlib.pyplot as plt
import math

def plot(archive, express, domain, config):
    fig = plt.figure() 
    if domain['plotscale']:
        scale = 100 #math.sqrt(config['resolution'])
    else:
        scale = 1
    phenotypes = express.do(archive['genes'], domain)
    for i in range(archive['genes'].shape[0]):
        phenotype = phenotypes[i]
        if phenotype is not None:
            dx = archive['features'][i,0]
            dy = archive['features'][i,1]
            fitness = archive['fitness'][i][0]
            if fitness>1.0:
                fitness = 1.0
            elif fitness<0.0:
                fitness = 0.0
            # express.visualize_raw(phenotype, [1-fitness, fitness, 0], i%20, i/20)
            express.visualize_raw(phenotype, [1-fitness, fitness, 0], dx*scale, dy*scale)
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    plt.xlim([0, scale])
    plt.ylim([0, scale])
    plt.axis('equal')
    # plt.show()
    return plt
