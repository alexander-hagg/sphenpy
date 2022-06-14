import yaml
import time

from sphen import sphen
from qd.voronoielites import visualize, visualize_phenotypes
from domain.simpleshapes import init, fitness_fun, express


config = yaml.safe_load(open("sphen/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

start = time.time()
archive = sphen.evolve(random_pop, config, domain, fitness_fun)
print(f'Time elapsed: {time.time() - start:.2}s.')

# visualize.plot(archive, domain)
qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
figure = visualize_phenotypes.plot(archive, express, domain, qdconfig)
figure.savefig('results/SPHEN prediction', dpi=600)

# Check ground truth
fitness, features = fitness_fun.get(archive['genes'], domain)
import qd.voronoielites.create_archive as ca
import qd.voronoielites.update_archive as update
archive_true = ca.create_archive(domain, qdconfig)
archive_true = update.update_archive(archive_true, archive['genes'], features, fitness, qdconfig, domain)

figure = visualize_phenotypes.plot(archive_true, express, domain, qdconfig)
figure.savefig('results/SPHEN ground truth', dpi=600)

