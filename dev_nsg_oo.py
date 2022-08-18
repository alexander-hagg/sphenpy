import yaml
import time
import numpy as np
import copy

from domain.ootest import init, fitness_function
from qd.mapelites import evolution, visualize, visualize_phenotypes


config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

start = time.time()
fitfun = lambda x: fitness_function.get(x, domain)
archive = evolution.evolve(random_pop, config, domain, fitfun)
print(f'Time elapsed: {time.time() - start:.2}s.')
archive.plot()

# fitness, features, phenotypes = fitness_fun.get(archive_geneslist, domain)
# plt = express.visualize_pyvista(phenotypes, domain, features, fitness)
