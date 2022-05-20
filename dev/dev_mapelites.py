import yaml
import time
import matplotlib.pyplot as plt

from qd.mapelites import qd, visualize, visualize_phenotypes
# from domain.rastrigin import init, fitness_fun, express
from domain.simpleshapes import init, fitness_fun, express

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

# Run QD, time it and visualize
start = time.time()
archive = qd.evolve(random_pop, config, domain, fitness_fun)
print(f'Time elapsed: {time.time() - start:.2}s.')
# visualize.plot(archive, domain)
visualize_phenotypes.plot(archive, express, domain, config)

