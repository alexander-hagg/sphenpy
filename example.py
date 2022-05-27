import yaml
import time
import pickle

from sphen import sphen
from qd.voronoielites import visualize, visualize_phenotypes
from domain.simpleshapes import init, fitness_fun, express
# from domain.rastrigin import init, fitness_fun, express


# Load SPHEN and domain configuration, including first population
config = yaml.safe_load(open("sphen/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

# Evolve archive with SPHEN
start = time.time()
archive = sphen.evolve(random_pop, config, domain, fitness_fun)
print(f'Time elapsed: {time.time() - start:.2}s.')

with open('archive_sphen.pkl', 'wb') as f:
    pickle.dump(archive, f)

# Visualization of archive (fitness and phenotypes)
figure = visualize.plot(archive, domain)
figure.savefig('results/SPHEN predicted fitness', dpi=600)
qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
figure = visualize_phenotypes.plot(archive, express, domain, qdconfig)
figure.savefig('results/SPHEN predicted phenotypes', dpi=600)
