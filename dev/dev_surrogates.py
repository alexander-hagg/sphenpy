import yaml
import time
import sys
import numpy as np

sys.path.insert(1, '/home/alex/sphenpy')
from sphen import sphen, surrogates
from qd.voronoielites import visualize, visualize_phenotypes
from domain.simpleshapes import init, fitness_fun, express


config = yaml.safe_load(open("sphen/config.yml"))
domain, samples = init.do(config.get('init_samples'))

fitness, features = fitness_fun.get(samples, domain)
targets = np.hstack([fitness, features])
models = surrogates.train(samples, targets[np.newaxis].T)
model_multiout = surrogates.train_multioutput(samples, targets)
