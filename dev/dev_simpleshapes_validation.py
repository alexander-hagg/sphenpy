import yaml
import time
import sys 
sys.path.insert(1, '/home/alex/sphenpy')
from sphen import sphen
from qd.voronoielites import visualize, visualize_phenotypes
from domain.simpleshapes import init, fitness_fun, express


domain, random_pop = init.do(1)

fitness, features = fitness_fun.get(random_pop, domain)

phenotypes = express.do(random_pop, domain)
plt = express.visualize_raw(phenotypes[0])
plt.show()
# express.visualize(phenotypes[0])