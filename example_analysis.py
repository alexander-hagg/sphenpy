import yaml
import pickle
import matplotlib.pyplot as plt
from qd.voronoielites import visualize, visualize_phenotypes
from domain.simpleshapes import init, fitness_fun, express


# Load SPHEN and domain configuration, including first population
config = yaml.safe_load(open("sphen/config.yml"))
domain, random_pop = init.do(config.get('init_samples'))

with open('archive_sphen.pkl', 'rb') as f:
    archive = pickle.load(f)

# Visualization of archive (fitness and phenotypes)
figure = visualize.plot(archive, domain)
figure.savefig('results/SPHEN predicted fitness', dpi=600)
plt.clf()
qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
figure = visualize_phenotypes.plot(archive, express, domain, qdconfig)
figure.savefig('results/SPHEN predicted phenotypes', dpi=600)
plt.clf()

# Check ground truth
fitness, features = fitness_fun.get(archive['genes'], domain)
import qd.voronoielites.create_archive as ca
import qd.voronoielites.update_archive as update
archive_true = ca.create_archive(domain, qdconfig)
archive_true = update.update_archive(archive_true, archive['genes'], features, fitness, qdconfig, domain)

with open('archive_true.pkl', 'wb') as f:
    pickle.dump(archive_true, f)

figure = visualize.plot(archive_true, domain)
figure.savefig('results/SPHEN true fitness', dpi=600)
plt.clf()
figure = visualize_phenotypes.plot(archive_true, express, domain, qdconfig)
figure.savefig('results/SPHEN true phenotypes', dpi=600)
