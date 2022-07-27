import yaml
import time
import numpy as np
import copy

from domain.nsg_cppn import init, fitness_fun, express, cppn

ninitsamples = 1
domain, random_pop = init.do(ninitsamples)

random_network = cppn.random()
# domain['genome_size'] = cppn.get_genome_sizes(random_network)
fitness, features, phenotypes = fitness_fun.get([random_network], domain)
# network = copy.deepcopy(random_network)
# cppn.mutate(network, probability=0.1, sigma=1.0)
# phenotypes = express.do([random_network, network], domain)
plt = express.visualize(phenotypes[0], domain, features[0])
# plt = express.visualize(phenotypes[1], domain)

# fitness, features = fitness_fun.get(random_pop, domain)

