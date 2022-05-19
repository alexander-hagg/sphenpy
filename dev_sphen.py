import yaml
import time
import matplotlib.pyplot as plt

from sphen import sphen
from qd.voronoielites import visualize, visualize_phenotypes
# from domain.rastrigin import init, fitness_fun, express
from domain.simpleshapes import init, fitness_fun, express

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("sphen/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

# Run QD, time it and visualize
start = time.time()
archive = sphen.evolve(random_pop, config, domain, fitness_fun)
# visualize.plot(archive, domain)
qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
visualize_phenotypes.plot(archive, express, domain, qdconfig)
print(f'Time elapsed: {time.time() - start:.2}s.')
