import GPy
from IPython.display import display
import numpy as np


def train(observation, targets):
    models = []
    for i in range(len(targets)):
        kernel = GPy.kern.RBF(input_dim=observation.shape[1], variance=0.1, lengthscale=0.1)
        m = GPy.models.GPRegression(observation,targets[i],kernel)
        m.optimize_restarts(messages=False, num_restarts = 3)
        models.append(m)
    return models

def predict(observation, model):
    mu, sigma = model.predict(observation)
    return mu

def ucb(observation, models, exploration_factor):
    mu0, sigma0 = models[0].predict(observation)
    # print(f'Mean mu: {np.mean(mu0)}, mean sigma0: {np.mean(sigma0)}')
    mu1, sigma = models[1].predict(observation)
    mu2, sigma = models[2].predict(observation)

    ucb = mu0 + exploration_factor * sigma0
    features = np.transpose(np.squeeze(np.asarray([mu1, mu2])))
    return ucb, features
