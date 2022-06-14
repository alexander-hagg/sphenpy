import numpy as np
import yaml
from scipy.stats import qmc

import sphen.surrogates as surrogates
from qd.mapelites import qd, visualize
from scipy.spatial.distance import cdist


def evolve(samples, config, domain, ff):
    # Get true values for sample set
    observation = samples
    fitness, features = ff.get(samples, domain)
    features = features[:,domain['features']]
    total_samples = config['total_samples']

    # Setup internal QD algorithm
    qdconfig = yaml.safe_load(open("qd/mapelites/config.yml"))

    # Setup Sobol sampler
    sampler = qmc.Sobol(d=len(domain['features']), scramble=True)

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
        # figure = visualize.plot(archive, domain)
        # figure.savefig('results/ucb_fitness_' + str(observation.shape[0]), dpi=600)

        # Select infill samples
        # Create emitter points using Sobol sequence and select closest points in the set (feature-based)
        emitter_points = sampler.random(config['num_samples'])
        distances = cdist(emitter_points, features)
        selection = np.argmin(distances, axis=1)
        sel_observation = observation[selection]

        # Get ground truth fitness and add samples to the observation set
        sel_fitness, sel_features = ff.get(sel_observation, domain)
        sel_features = sel_features[:,domain['features']]
        observation = np.vstack([observation, sel_observation])
        fitness = np.vstack([fitness, sel_fitness])
        features = np.vstack([features, sel_features])

    # Evolve archive with acquisition function
    ucbfitfun = lambda x: surrogates.ucb(x, models, 0)
    archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)

    return archive