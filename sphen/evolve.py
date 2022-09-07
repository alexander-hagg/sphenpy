import numpy as np
from scipy.stats import qmc
import importlib

import sphen.surrogates as surrogates
from scipy.spatial.distance import cdist


def evolve(samples, config, qdconfig, domain, ff):
    qd = importlib.import_module('qd.' + qdconfig['algorithm'] + '.evolution')
    visualize = importlib.import_module('qd.' + qdconfig['algorithm'] + '.visualize')

    observation = samples
    fitness, features = ff.get(observation, domain)
    features = features[:,domain['features']]
    total_samples = config['total_samples']

    # Setup sampler
    sampler = qmc.Sobol(d=len(domain['features']), scramble=True)

    # This is the main sampling loop
    while observation.shape[0] <= total_samples:
        print(f'Current samples: {observation.shape[0]}/{total_samples}')

        # Train surrogate models
        models = surrogates.train(observation, [fitness, features[:,0][np.newaxis].T, features[:,1][np.newaxis].T])

        if observation.shape[0] >= total_samples:
            break

        # Evolve archive with acquisition function
        ucbfitfun = lambda x: surrogates.ucb(x, models, config['exploration_factor'])
        archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        if config['visualize_intermediate'] is True:
            figure = visualize.plot(archive, domain, ucbplot=True)
            figure.savefig('results/ucb_fitness_' + str(observation.shape[0]), dpi=600)

        # Flatten genes and features (due to MAP-Elites-like structure of some archives)
        flat_genes = archive['genes'].reshape((-1, domain['dof']))
        flat_genes = flat_genes[~np.isnan(flat_genes).any(axis=1)]
        flat_features = archive['features'].reshape((-1, len(domain['features'])))
        flat_features = flat_features[~np.isnan(flat_features).any(axis=1)]

        # Select infill samples
        # Create emitter points in the archive using Sobol sequence and select closest archive members
        emitter_points = sampler.random(config['num_samples'])
        distances = cdist(emitter_points, flat_features)
        selection = np.argmin(distances, axis=1)
        sel_observation = flat_genes[selection]

        # Get ground truth fitness and add samples to the observation set
        sel_fitness, sel_features = ff.get(sel_observation, domain)
        sel_features = sel_features[:,domain['features']]
        observation = np.vstack([observation, sel_observation])
        fitness = np.vstack([fitness, sel_fitness])
        features = np.vstack([features, sel_features])

    # Finally, evolve an archive based on fitness alone (no more exploring)
    ucbfitfun = lambda x: surrogates.ucb(x, models, 0)
    archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)

    return archive
