import numpy as np

def mapelites():
    # Initialization
    random_pop = np.random.rand(config.get('init_samples'),domain.get('dof'))
    archive = create_archive(domain, config)
    fitness, features = fitness_fun(random_pop, domain)
    replaced, replacement = niche_compete(fitness, features, archive, domain, config)
    archive = update_map(replaced, replacement, archive, fitness, random_pop, features)

    # Evolution
    for iGen in range(config.get('num_gens')):
        print('Generation: ' + str(iGen) + '/' + str(config.get('num_gens')))
        children = np.array([])
        while children.shape[0] < config.get('num_children'):
            new_children = create_children(archive, domain, config)
            children = np.vstack([children, new_children]) if children.size else new_children

        fitness, features = fitness_fun(children, domain)
        replaced, replacement = niche_compete(fitness, features, archive, domain, config)
        archive = update_map(replaced, replacement, archive, fitness, children, features)

    return archive

