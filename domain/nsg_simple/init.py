import numpy as np
import math
import yaml
from scipy.stats import qmc


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/nsg_simple/domain.yml"))

    domain['par_ranges'] = [domain['par_ranges'][0] * np.ones(domain['dof']), domain['par_ranges'][1] * np.ones(domain['dof'])]
    
    sampler = qmc.Sobol(d=domain['dof'], scramble=True)
    random_pop = sampler.random(ninit_samples)
    # Project sampling to parameter ranges
    ranges = np.asarray(domain['par_ranges'])
    random_pop = np.multiply(random_pop, (ranges[1] - ranges[0])) + ranges[0]
    # random_pop = np.floor(random_pop)

    return domain, random_pop
