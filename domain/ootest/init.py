import numpy as np
import yaml

from domain.ootest.genome import genome

def do(ninit_samples):
    domain = yaml.safe_load(open("domain/ootest/domain.yml"))

    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(genome(domain))

    return domain, random_pop
