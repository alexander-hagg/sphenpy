import numpy as np
import yaml

from domain.rastrigin.genome import math_genome

def do(ninit_samples):
    domain = yaml.safe_load(open("domain/rastrigin/domain.yml"))

    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(math_genome(domain))

    return domain, random_pop
