import GPy
from IPython.display import display

def train(observation, targets):
    # GPy.plotting.change_plotting_library('plotly')
    models = []
    for i in range(len(targets)):
        kernel = GPy.kern.RBF(input_dim=observation.shape[1], variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(observation,targets[i],kernel)
        m.optimize_restarts(messages=False, num_restarts = 10)
        models.append(m)
    return models

def predict(observation, model):
    mu, sigma = model.predict(observation)
    return mu

def ucb(observation, models, w):
    mu0, sigma0 = models[0].predict(observation)
    mu1, sigma = models[1].predict(observation)
    mu2, sigma = models[2].predict(observation)

    ucb = mu0 + w * sigma0
    return ucb, mu1, mu2
