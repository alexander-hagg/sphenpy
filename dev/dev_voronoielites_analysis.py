import yaml
import time
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.insert(1, '/home/alex/sphenpy')
from qd.voronoielites import qd, visualize, visualize_phenotypes
# from domain.rastrigin import init, fitness_fun, express
from domain.simpleshapes import init, fitness_fun, express

# Configuration of QD, domain and initial population
config = yaml.safe_load(open("qd/voronoielites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

# Create lambda function for fitness (necessary due to multiple contexts of use (in SPHEN, and pure QD))
fitfun = lambda x: fitness_fun.get(x, domain)

with open('archive_ve.pkl', 'rb') as f:
    archive = pickle.load(f)

# Visualization of archive (fitness and phenotypes)
# figure = visualize.plot(archive, domain)
# figure.savefig('results/VE fitness', dpi=600)
plt.clf()
figure = visualize_phenotypes.plot(archive, express, domain, config)
figure.savefig('results/VE phenotypes', dpi=600, bbox_inches='tight')
plt.clf()

# phenotypes = express.do(archive['genes'], domain)

# for i in range(400):
    # express.visualize_raw(phenotypes[i], [0, 0, 0], i%20, i/20)
    # plt.title(i)
    # plt.xlim(-1, 21)
    # plt.ylim(-1, 21)
    # plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
