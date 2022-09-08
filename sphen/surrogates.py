import GPy
import numpy as np


def train(observation, targets):
    models = []
    flat = observation_flatten(observation)
    for i in range(len(targets)):
        kernel = GPy.kern.RBF(input_dim=flat.shape[1], variance=0.1, lengthscale=0.1)
        m = GPy.models.GPRegression(flat, targets[i].T, kernel)
        m.optimize_restarts(optimizer='bfgs', messages=False, num_restarts=10, verbose=False)
        models.append(m)
    return models


def observation_flatten(observation):
    # input_dim=observation[0].domain['dof']
    num_observations = len(observation)
    flat = []
    for i in range(num_observations):
        flat.append(observation[i].get_genome())
    flat = np.concatenate(flat, axis=1).T
    return flat


def predict(observation, model):
    flat = observation_flatten(observation)
    mu, sigma = model.predict(flat)
    return mu


def ucb(observation, models, exploration_factor):
    flat = observation_flatten(observation)
    mu0, sigma0 = models[0].predict(flat)
    mu1, sigma = models[1].predict(flat)
    mu2, sigma = models[2].predict(flat)

    ucb = mu0 + exploration_factor * sigma0
    features = np.squeeze(np.asarray([mu1, mu2])).T
    return ucb.T, features


def train_sparse(observation, targets):
    models = []
    for i in range(len(targets)):
        print(i)
        # kernel = GPy.kern.RBF(input_dim=observation.shape[1], variance=0.1, lengthscale=0.1)
        Z = np.random.rand(10,observation.shape[1])
        m = GPy.models.SparseGPRegression(observation,targets[i],Z=Z)
        m.optimize_restarts(optimizer='bfgs', messages=False, num_restarts=10, verbose=False)
        models.append(m)
    return models


def train_multioutput(observation, targets):
    kernel = GPy.kern.RBF(observation.shape[1],lengthscale=0.1)**GPy.kern.Coregionalize(input_dim=observation.shape[1],output_dim=3, rank=1)
    m = GPy.models.MultioutputGP(observation,targets,kernel=kernel)
    m.optimize_restarts(messages=False, num_restarts=10, verbose=False)
    return m
