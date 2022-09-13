import numpy as np
from scipy.stats import qmc
import importlib

import sphen.surrogates as surrogates
from scipy.spatial.distance import cdist


def evolve(samples, config, qdconfig, domain, ff):
    qd = importlib.import_module('qd.' + qdconfig['algorithm'] + '.evolution')
    # visualize = importlib.import_module('qd.' + qdconfig['algorithm'] + '.visualize')

    observation = samples
    fitness, features = ff(observation)[0:2]
    features = features[:,domain['features']]
    total_samples = config['total_samples']

    # Setup sampler
    sampler = qmc.Sobol(d=len(domain['features']), scramble=True)

    # This is the main sampling loop
    while len(observation) <= total_samples:
        print(f'Current samples: {len(observation)}/{total_samples}')

        # Train surrogate models
        models = surrogates.train(observation, [fitness, features[:,0][np.newaxis], features[:,1][np.newaxis]])

        if len(observation) >= total_samples:
            break

        # Evolve archive with acquisition function
        ucbfitfun = lambda x: surrogates.ucb(x, models, config['exploration_factor'])
        archive, improvement = qd.evolve(observation, qdconfig, domain, ucbfitfun)
        if config['visualize_intermediate'] is True:
            figure = visualize.plot(archive, domain, ucbplot=True)
            figure.savefig('results/ucb_fitness_' + str(observation.shape[0]), dpi=600)

        # Flatten genes and features (due to MAP-Elites-like structure of some archives)
        flat_genes = archive.genes
        flat_genes = flat_genes.reshape((-1, 1))
        flat_features = archive.features
        flat_features = flat_features.reshape((-1, len(domain['features'])))
        flat_fitness = archive.fitness
        flat_fitness = flat_fitness.reshape((-1, 1))
        present = np.squeeze(~np.isnan(flat_fitness))
        flat_genes = flat_genes[present]
        flat_fitness = flat_fitness[present,:]
        flat_features = flat_features[present,:]

        # Select infill samples
        # Create emitter points in the archive using Sobol sequence and select closest archive members
        emitter_points = sampler.random(config['num_samples'])
        distances = cdist(emitter_points, flat_features)
        selection = np.argmin(distances, axis=1)
        sel_observation = np.squeeze(flat_genes[selection])

        # Get ground truth fitness and add samples to the observation set
        sel_fitness, sel_features = ff(sel_observation)[0:2]
        sel_features = sel_features[:,domain['features']]

        observation.extend(sel_observation)
        fitness = np.hstack([fitness, sel_fitness])
        features = np.vstack([features, sel_features])

    # Finally, evolve an archive based on fitness alone (no more exploring)
    ucbfitfun = lambda x: surrogates.ucb(x, models, 0)
    archive = qd.evolve(observation, qdconfig, domain, ucbfitfun)

    return archive
