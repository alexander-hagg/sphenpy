import matplotlib.pyplot as plt

def plot(archive, express, domain, config):
    scale = 2*config['resolution']
    for i in range(archive['genes'].shape[0]):
        for j in range(archive['genes'].shape[1]):
            genome = archive['genes'][i,j,:]
            phenotype = express.express_single(genome, domain)
            if phenotype is not None:
                dx = archive['features'][i,j,0]
                dy = archive['features'][i,j,1]
                fitness = archive['fitness'][i,j]
                express.visualize_raw([phenotype[0]+dx*scale, phenotype[1]+dy*scale], [1-fitness, fitness, 0])
    plt.axis('equal')
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    plt.show()
