import numpy as np
import yaml

from domain.rastrigin.genome import MathGenome


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/rastrigin/domain.yml"))

    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(MathGenome(domain))

    return domain, random_pop
