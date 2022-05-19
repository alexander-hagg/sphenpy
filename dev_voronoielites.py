import yaml
import time
import matplotlib.pyplot as plt

from qd.voronoielites import qd, visualize, visualize_phenotypes
# from domain.rastrigin import init, fitness_fun, express
from domain.simpleshapes import init, fitness_fun, express

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("qd/voronoielites/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

# Run QD, time it and visualize
timeit = False
start = time.time()
if timeit:
    import cProfile
    cProfile.run('qd.evolve(random_pop, config, domain, fitness_fun)', 'restats')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('restats')
    p.sort_stats(SortKey.TIME).print_stats(10)
else:
    archive = qd.evolve(random_pop, config, domain, fitness_fun)
    visualize.plot(archive, domain)
    visualize_phenotypes.plot(archive, express, domain, config)

print(f'Time elapsed: {time.time() - start:.2}s.')
