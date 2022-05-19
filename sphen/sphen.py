import numpy as np
import yaml
import GPy
from IPython.display import display

import sphen.surrogates as surrogates
from qd.voronoielites import qd

def evolve(samples, config, domain, ff):
    # Get true values for sample set
    observation = samples
    fitness, features = ff.get(samples, domain)
    total_samples = config['total_samples']

    # Setup internal QD algorithm
    qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
    
    # Sampling loop
    while observation.shape[0] <= total_samples:
        print(f'Current samples: {observation.shape[0]}/{total_samples}')

        # Train surrogate models
        models = surrogates.train(observation, [fitness, features[:,0][np.newaxis].T, features[:,1][np.newaxis].T])
        
        # Evolve archive with acquisition function
        ucbfitfun = lambda x: surrogates.ucb(x, models, config['exploration_factor'])        
        archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        
        # Select infill samples
        selection = np.random.randint(0, high=archive['genes'].shape[0], size=config['num_samples'])
        sel_observation = archive['genes'][selection]
        sel_fitness, sel_features = ff.get(sel_observation, domain)
        observation = np.vstack([observation, sel_observation])
        fitness = np.vstack([fitness, sel_fitness])
        features = np.vstack([features, sel_features])
    
    # Evolve archive with acquisition function
    ucbfitfun = lambda x: surrogates.ucb(x, models, 0) 
    archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        
    return archive