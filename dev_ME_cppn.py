import time
import matplotlib.pyplot as plt

from qd.mapelites import evolution, cfg
# from qd.voronoielites import evolution, cfg

# from domain.rastrigin import init, fitness_function
from domain.nsg_cppn import init, fitness_function, plotgrid

config = cfg.get()
config['num_gens'] = 5000
config['init_samples'] = 1000
config['resolution'] = 10


domain, random_pop = init.do(config['init_samples'])

start = time.time()
fitfun = lambda x: fitness_function.get(x, domain)

archive,improvement = evolution.evolve(random_pop, config, domain, fitfun)

print(f'Time elapsed: {time.time() - start:.2}s.')
# archive.plot()

list_genomes = archive.create_pool()
fitness, features, phenotypes, rawfeatures = fitfun(list_genomes)
niches = archive.get_niches()
output_visualization_file = 'archive.png'
plot = plotgrid.plot(phenotypes, domain, features=features, fitness=fitness, niches=niches, rawfeatures=rawfeatures, filename=output_visualization_file, gridresolution=config['resolution'], output_resolution=[8000, 8000])
