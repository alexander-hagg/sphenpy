import yaml
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/alex/sphenpy')
from sphen import sphen, surrogates
from qd.voronoielites import visualize, visualize_phenotypes
from domain.simpleshapes import init, fitness_fun, express


config = yaml.safe_load(open("sphen/config.yml"))
domain, samples = init.do(config.get('init_samples'))

fitness, features = fitness_fun.get(samples, domain)
targets = np.hstack([fitness, features])
models = surrogates.train(samples, targets[np.newaxis].T)
models_sparse = surrogates.train_sparse(samples, targets[np.newaxis].T)

# domain, samples = init.do(config.get('init_samples'))

y0 = []
for i in range(3):
    y0.append(surrogates.predict(samples, models[i]))
y0 = np.asarray(y0)
y0 = y0.T
y0 = np.squeeze(y0)

y1 = []
for i in range(3):
    y1.append(surrogates.predict(samples, models_sparse[i]))
y1 = np.asarray(y1)
y1 = y1.T
y1 = np.squeeze(y1)

print(y0)
print(y1)
plt.clf()

plt.scatter(samples[:,1], y0[:,0], c=[0, 0, 0])
plt.scatter(samples[:,1], y1[:,0], c=[1, 0, 0])
plt.show()
