import numpy as np
import yaml


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/rastrigin/domain.yml"))
    random_pop = np.random.rand(ninit_samples,domain['dof'])
    # Project sampling to parameter ranges
    ranges = np.asarray(domain['par_ranges'])
    random_pop = np.multiply(random_pop, (ranges[1] - ranges[0])) + ranges[0]
    return domain, random_pop
