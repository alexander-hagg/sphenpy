import GPy
import numpy as np
from IPython.display import display
import sphen.surrogates as surrogates


def evolve(samples, config, domain, ff):
    # Get true values for sample set
    observation = samples
    fitness, features = ff.get(samples, domain)

    # Train surrogate models
    models = surrogates.train(observation, [fitness, features[:,0][np.newaxis].T, features[:,1][np.newaxis].T])
    
    ucb, mu1, mu2 = surrogates.ucb(observation, models, config['ucb'])
    print(ucb)