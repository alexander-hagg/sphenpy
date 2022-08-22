import yaml
import time
import matplotlib.pyplot as plt

from qd.mapelites import evolution

# from domain.rastrigin import init, fitness_function
from domain.nsg_cppn import init, fitness_function, plotgrid

import cProfile, pstats

config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

start = time.time()
fitfun = lambda x: fitness_function.get(x, domain)

# profiler = cProfile.Profile()
# profiler.enable()
archive,improvement = evolution.evolve(random_pop, config, domain, fitfun)
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
# stats.print_stats()

print(f'Time elapsed: {time.time() - start:.2}s.')
# archive.plot()

list_genomes = archive.create_pool()
fitness, features, phenotypes, rawfeatures = fitfun(list_genomes)
niches = archive.get_niches()
plot = plotgrid.plot(phenotypes, domain, features=features, fitness=fitness, niches=niches, rawfeatures=rawfeatures, filename='archive.png', gridresolution=config['resolution'], output_resolution=[4000, 4000])
