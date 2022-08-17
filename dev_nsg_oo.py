import yaml
import time
import numpy as np
import copy

from domain.ootest import init, fitness
from qd.mapelites import evolution, visualize, visualize_phenotypes



config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

print(f'random_pop: \n{random_pop}')

phenotypes = [x.express() for x in random_pop]

print(f'phenotypes: {phenotypes}')

[x.mutate(config['mut_probability'], config['mut_sigma']) for x in random_pop]
phenotypes = [x.express() for x in random_pop]

print(f'phenotypes: {phenotypes}')



fitness, features, phenotypes = fitness.get(random_pop, domain)

# plt = express.visualize_pyvista(phenotypes, domain, features)

# start = time.time()
# fitfun = lambda x: fitness_fun.get(x, domain)
# config = yaml.safe_load(open("qd/mapelites/config.yml"))
# domain, random_pop = init.do(config['init_samples'])

# archive = evolution.evolve(random_pop, config, domain, fitfun)
# print(f'Time elapsed: {time.time() - start:.2}s.')
# visualize.plot(archive, domain)

# archive_geneslist = np.squeeze((archive['genes'])).flatten().tolist()
# # print(archive_geneslist)
# fitness, features, phenotypes = fitness_fun.get(archive_geneslist, domain)
# plt = express.visualize_pyvista(phenotypes, domain, features, fitness)
# # visualize_phenotypes.plot(archive, express, domain, config)
