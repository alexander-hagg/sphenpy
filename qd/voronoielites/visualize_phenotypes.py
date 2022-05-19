import matplotlib.pyplot as plt

def plot(archive, express, domain, config):
    fig = plt.figure() 
    if domain['plotscale']:
        scale = 0.5*config['resolution']
    else:
        scale = 1
    for i in range(archive['genes'].shape[0]):
        genome = archive['genes'][i,:]
        phenotype = express.express_single(genome, domain)
        if phenotype is not None:
            dx = archive['features'][i,0]
            dy = archive['features'][i,1]
            fitness = archive['fitness'][i][0]
            # express.visualize_raw([phenotype[0]+dx*scale, phenotype[1]+dy*scale], [1-fitness, fitness, 0])
            express.visualize_raw(phenotype, [1-fitness, fitness, 0], dx*scale, dy*scale)
    plt.axis('equal')
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    plt.show()
