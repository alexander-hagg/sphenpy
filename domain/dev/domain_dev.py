config = {
    'resolution': 4,
    'num_gens': 50,
    'num_children': 4,
    'mut_sigma': 0.1,
    'init_samples': 3,
}

domain = {
    'name': 'debug',
    'features': [1,2],
    'dof': 6,
    'nfeatures': 2,
    'par_ranges': np.array([[0, 0, 0, 0],[100, 100, 5, 5]]),
    'fit_range': np.array([-5, 5]),
}

def fitness_fun(population, domain):
    fitness = np.random.rand(1,population.shape[0])
    fitness[fitness > 1] = 1.
    fitness[fitness < 0] = 0.
    features = np.random.rand(domain.get('nfeatures'),population.shape[0])
    features[features > 1] = 1.
    features[features < 0] = 0.
    return fitness, features
