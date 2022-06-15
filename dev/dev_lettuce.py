import yaml
import time
import pickle

from sphen import sphen
from qd.voronoielites import visualize, visualize_phenotypes
# from domain.rastrigin import init, fitness_fun, express
# from domain.simpleshapes import init, fitness_fun, express
from domain.lettuce2d import init, fitness_fun, express


# Load SPHEN and domain configuration, including first population
config = yaml.safe_load(open("sphen/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

phenotypes = express.do(random_pop, domain)

fitness, features = fitness_fun.get(random_pop, domain)
print(f'fitness: {fitness}')
print(f'features: {features}')

