import yaml
import time
import numpy as np

from domain.nsg_cppn import init, fitness_fun, express, cppn

ninitsamples = 1
domain, random_pop = init.do(ninitsamples)
# print(random_pop)

# phenotypes = express.do_surf(random_pop, domain)
# plt = express.visualize_surf(phenotypes[0], domain)

# phenotypes = express.do(random_pop, domain)
# plt = express.visualize(phenotypes[0], domain)

phenotypes = express.do(random_pop, domain)
for i in range(ninitsamples):
    plt = express.visualize(phenotypes[i], domain)
    # time.sleep(1)

# input = np.array([1, 1])
# print("Get CPPN output")
# output = cppn.forward(input, net)
# print(output)


# fitness, features = fitness_fun.get(random_pop, domain)

