import numpy as np
import yaml

from domain.ootest.genome import cppn_genome

def do(ninit_samples):
    domain = yaml.safe_load(open("domain/ootest/domain.yml"))

    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(cppn_genome(domain))

    return domain, random_pop
