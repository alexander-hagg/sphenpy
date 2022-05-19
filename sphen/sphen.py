import numpy as np
import yaml
from scipy.stats import qmc

import sphen.surrogates as surrogates
from qd.voronoielites import qd
from scipy.spatial.distance import cdist


def evolve(samples, config, domain, ff):
    # Get true values for sample set
    observation = samples
    fitness, features = ff.get(samples, domain)
    total_samples = config['total_samples']

    # Setup internal QD algorithm
    qdconfig = yaml.safe_load(open("qd/voronoielites/config.yml"))
    
    # Setup Sobol sampler
    sampler = qmc.Sobol(d=domain['nfeatures'], scramble=True)
            
    # Sampling loop
    while observation.shape[0] <= total_samples:
        print(f'Current samples: {observation.shape[0]}/{total_samples}')

        # Train surrogate models
        models = surrogates.train(observation, [fitness, features[:,0][np.newaxis].T, features[:,1][np.newaxis].T])
        
        if observation.shape[0] >= total_samples:
            break

        # Evolve archive with acquisition function
        ucbfitfun = lambda x: surrogates.ucb(x, models, config['exploration_factor'])        
        archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        
        # Select infill samples
        # Create emitter points using Sobol sequence and select closest points in the set (feature-based)
        emitter_points = sampler.random(config['num_samples'])
        distances = cdist(emitter_points, features)
        selection = np.argmin(distances, axis=1)
        
        sel_observation = archive['genes'][selection]
        sel_fitness, sel_features = ff.get(sel_observation, domain)
        observation = np.vstack([observation, sel_observation])
        fitness = np.vstack([fitness, sel_fitness])
        features = np.vstack([features, sel_features])
    
    # Evolve archive with acquisition function
    ucbfitfun = lambda x: surrogates.ucb(x, models, 0) 
    archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        
    return archive