import yaml
import time

from qd.mapelites import qd, visualize
# from domain.random import init, fitness_fun
from domain.rastrigin import init, fitness_fun

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config)

# Run QD and visualize
start = time.time()
archive = qd.evolve(random_pop, config, domain, fitness_fun)
end = time.time()
print(f'Time elapsed: {end - start:.2}s.')
visualize.plot(archive)
