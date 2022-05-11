import numpy as np
import yaml

def do(ninit_samples):
    domain = yaml.safe_load(open("domain/random/domain.yml"))
    random_pop = np.random.rand(ninit_samples,domain['dof'])
    return domain, random_pop