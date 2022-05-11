import yaml
import time

from qd.mapelites import qd, visualize
from domain.simpleshapes import init, fitness_fun, express

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

# phenotypes = express.do(random_pop, domain)
# print(phenotypes[0])

fitness_fun.get(random_pop, domain)