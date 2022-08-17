import yaml
import time
import numpy as np
import copy

from domain.nsg_cppn import init, fitness_fun, express, cppn
from qd.mapelites import evolution, visualize, visualize_phenotypes

# from domain.nsg_cppn.cppn import cppn
# test = np.empty((30,30), dtype=cppn)
# print(test)

# fitness, features, phenotypes = fitness_fun.get(random_pop, domain)
# plt = express.visualize_pyvista(phenotypes, domain, features)

start = time.time()
fitfun = lambda x: fitness_fun.get(x, domain)
config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

archive = evolution.evolve(random_pop, config, domain, fitfun)
print(f'Time elapsed: {time.time() - start:.2}s.')
# visualize.plot(archive, domain)

archive_geneslist = np.squeeze((archive['genes'])).flatten().tolist()
# print(archive_geneslist)
fitness, features, phenotypes = fitness_fun.get(archive_geneslist, domain)
plt = express.visualize_pyvista(phenotypes, domain, features, fitness)
# visualize_phenotypes.plot(archive, express, domain, config)
