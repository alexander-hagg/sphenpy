import numpy as np
import yaml

def do(config):
    domain = yaml.safe_load(open("domain/random/domain.yml"))
    random_pop = np.random.rand(config.get('init_samples'),domain.get('dof'))
    return domain, random_pop