import yaml
import time
import numpy as np
import copy

from domain.nsg_cppn import init, fitness_fun, express, cppn
from qd.mapelites import evolution, visualize, visualize_phenotypes

ninitsamples = 3
domain, random_pop = init.do(ninitsamples)
# fitness, features, phenotypes = fitness_fun.get(random_pop, domain)
# plt = express.visualize_pyvista(phenotypes, domain, features)
# quit()

start = time.time()
fitfun = lambda x: fitness_fun.get(x, domain)
config = yaml.safe_load(open("qd/mapelites/config.yml"))

archive = evolution.evolve(random_pop, config, domain, fitfun)
print(f'Time elapsed: {time.time() - start:.2}s.')
# visualize.plot(archive, domain)
visualize_phenotypes.plot(archive, express, domain, config)

