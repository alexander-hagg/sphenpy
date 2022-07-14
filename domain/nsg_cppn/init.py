import numpy as np
import math
import yaml
from scipy.stats import qmc
from domain.nsg_cppn import cppn


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/nsg_cppn/domain.yml"))

    domain['dof'] = domain['num_units'] * domain['dof_perblock']
    domain['par_ranges'] = np.tile(domain['par_ranges'], math.floor(domain['num_units']))
    ranges = np.asarray(domain['par_ranges'])
    # sampler = qmc.Sobol(d=domain['dof'], scramble=True)
    # random_pop = sampler.random(ninit_samples)
    # Project sampling to parameter ranges
    # random_pop = np.multiply(random_pop, (ranges[1] - ranges[0])) + ranges[0]
    # random_pop = np.round(random_pop)
    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(cppn.random())

    print(f'random_pop.size: {len(random_pop)}')

    return domain, random_pop
