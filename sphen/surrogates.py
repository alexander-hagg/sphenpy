import GPy
import numpy as np


def train(observation, targets):
    models = []
    for i in range(len(targets)):
        kernel = GPy.kern.RBF(input_dim=observation.shape[1], variance=0.1, lengthscale=0.1)
        m = GPy.models.GPRegression(observation, targets[i], kernel)
        m.optimize_restarts(optimizer='bfgs', messages=False, num_restarts=10, verbose=False)
        models.append(m)
    return models


def predict(observation, model):
    mu, sigma = model.predict(observation)
    return mu


def ucb(observation, models, exploration_factor):
    mu0, sigma0 = models[0].predict(observation)
    mu1, sigma = models[1].predict(observation)
    mu2, sigma = models[2].predict(observation)

    ucb = mu0 + exploration_factor * sigma0
    features = np.transpose(np.squeeze(np.asarray([mu1, mu2])))
    return ucb, features


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
