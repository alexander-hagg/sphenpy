import numpy as np
import yaml
import GPy
from IPython.display import display

import sphen.surrogates as surrogates
from qd.voronoielites import qd

def evolve(samples, config, domain, ff):
    # Get true values for sample set
    num_samples = samples.shape[0]
    observation = samples
    fitness, features = ff.get(samples, domain)
    total_samples = config['total_samples']

    # Setup internal QD algorithm
    qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
    
    # Sampling loop
    while num_samples <= total_samples:
        print(f'Current samples: {num_samples}/{total_samples}')
        
        # Train surrogate models
        models = surrogates.train(observation, [fitness, features[:,0][np.newaxis].T, features[:,1][np.newaxis].T])
        
        # Create lambda function for fitness (necessary due to multiple contexts of use (in SPHEN, and pure QD))
        ucbfitfun = lambda x: surrogates.ucb(x, models)
        
        # Evolve archive with acquisition function
        archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        
        # Select infill samples
        