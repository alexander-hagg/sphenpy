import yaml
import time
# from domain.rastrigin import init, fitness_function
from domain.nsg_cppn import init, fitness_function, plotgrid
from qd.mapelites import evolution

config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain, random_pop = init.do(config['init_samples'])

start = time.time()
fitfun = lambda x: fitness_function.get(x, domain)
archive = evolution.evolve(random_pop, config, domain, fitfun)
print(f'Time elapsed: {time.time() - start:.2}s.')
# archive.plot()

list_genomes = archive.create_pool()
fitness, features, phenotypes = fitfun(list_genomes)
plotgrid.plot(phenotypes, domain, features, fitness)
