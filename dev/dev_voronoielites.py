import yaml
import time
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.insert(1, '/home/alex/sphenpy')
from qd.voronoielites import qd, visualize, visualize_phenotypes
from domain.rastrigin import init, fitness_fun, express
# from domain.simpleshapes import init, fitness_fun, express

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("qd/voronoielites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

# Create lambda function for fitness (necessary due to multiple contexts of use (in SPHEN, and pure QD))
fitfun = lambda x: fitness_fun.get(x, domain)

# Run QD, time it and visualize
start = time.time()
archive = qd.evolve(random_pop, config, domain, fitfun)
print(f'Time elapsed: {time.time() - start:.2}s.')

# with open('archive_ve.pkl', 'wb') as f:
#     pickle.dump(archive, f)

# Visualization of archive (fitness and phenotypes)
figure = visualize.plot(archive, domain)
figure.savefig('results/VE fitness', dpi=600)
# figure = visualize_phenotypes.plot(archive, express, domain, config)
# figure.savefig('results/VE phenotypes', dpi=600)
