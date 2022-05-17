import matplotlib.pyplot as plt

def do(genomes, domain):
    return genomes

def express_single(genome, domain):
    return genome

def visualize_raw(phenotype, color=[0, 0, 0]):
    plt.scatter(phenotype[0], phenotype[1], color=color)
    
def visualize(phenotype):
    visualize_raw(phenotype)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.show()
