import time
import matplotlib.pyplot as plt

from qd.mapelites import cfg as qdcfg
from sphen import evolution, cfg
from domain.nsg_cppn import init, fitness_function, plotgrid


qdconfig = qdcfg.get()
qdconfig["num_gens"] = 500
config = cfg.get()
config["total_samples"] = 500
domain, random_pop = init.do(config["init_samples"])

start = time.time()
fitfun = lambda x: fitness_function.get(x, domain)
archive, improvement = evolution.evolve(random_pop, config, qdconfig, domain, fitfun)
print(f"Time elapsed: {time.time() - start:.2}s.")
# Archive.plot()

list_genomes = archive.create_pool()
fitness, features, phenotypes, rawfeatures = fitfun(list_genomes)
niches = archive.get_niches()
plot = plotgrid.plot(
    phenotypes,
    domain,
    features=features,
    fitness=fitness,
    niches=niches,
    rawfeatures=rawfeatures,
    filename="archive.png",
    gridresolution=qdconfig["resolution"],
    output_resolution=[4000, 4000],
)
