import yaml
import time
import pickle
import importlib

from sphen import sphen
# from domain.rastrigin import init, fitness_fun, express
from domain.simpleshapes import init, fitness_fun, express
# from domain.lettuce2d import init, fitness_fun, express


# Load SPHEN and domain configuration, including first population
config = yaml.safe_load(open("sphen/config.yml"))
qdconfig = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, samples = init.do(config.get('init_samples'))

# Evolve archive with SPHEN
start = time.time()
archive = sphen.evolve(samples, config, qdconfig, domain, fitness_fun)
print(f'Time elapsed: {time.time() - start:.2}s.')

with open('archive_sphen_lettuce.pkl', 'wb') as f:
    pickle.dump(archive, f)

# Visualization of archive (fitness and phenotypes)
visualize = importlib.import_module('qd.' + qdconfig['algorithm'] + '.visualize')
visualize_phenotypes = importlib.import_module('qd.' + qdconfig['algorithm'] + '.visualize_phenotypes')
figure = visualize.plot(archive, domain)
figure.savefig('results/SPHEN predicted fitness', dpi=600)
figure = visualize_phenotypes.plot(archive, express, domain, qdconfig)
figure.savefig('results/SPHEN predicted phenotypes', dpi=600)
