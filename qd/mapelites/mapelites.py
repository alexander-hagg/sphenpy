import numpy as np
import create_archive as ca
import niche_compete as compete
import update_map as update
import create_children as cc

def mapelites(config, domain, ff):
    # Initialization
    random_pop = np.random.rand(config.get('init_samples'),domain.get('dof'))
    archive = ca.create_archive(domain, config)
    fitness, features = ff.fitness_fun(random_pop, domain)
    replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
    archive = update.update_map(replaced, replacement, archive, fitness, random_pop, features)

    # Evolution
    for iGen in range(config.get('num_gens')):
        print('Generation: ' + str(iGen) + '/' + str(config.get('num_gens')))
        children = np.array([])
        while children.shape[0] < config.get('num_children'):
            new_children = cc.create_children(archive, domain, config)
            children = np.vstack([children, new_children]) if children.size else new_children

        fitness, features = ff.fitness_fun(children, domain)
        replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
        archive = update.update_map(replaced, replacement, archive, fitness, children, features)

    return archive

