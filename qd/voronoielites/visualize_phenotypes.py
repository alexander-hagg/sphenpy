import matplotlib.pyplot as plt

def plot(archive, express, domain, config):
    if domain['plotscale']:
        scale = config['resolution']
    else:
        scale = 1
    for i in range(archive['genes'].shape[0]):
        genome = archive['genes'][i,:]
        phenotype = express.express_single(genome, domain)
        if phenotype is not None:
            dx = archive['features'][i,0]
            dy = archive['features'][i,1]
            fitness = archive['fitness'][i][0]
            express.visualize_raw([phenotype[0]+dx*scale, phenotype[1]+dy*scale], [1-fitness, fitness, 0])
    plt.axis('equal')
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    plt.show()
