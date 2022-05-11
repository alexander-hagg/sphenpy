import numpy as np
import qd.mapelites.create_archive as ca
import qd.mapelites.niche_compete as compete
import qd.mapelites.update_map as update
import qd.mapelites.create_children as cc

def evolve(init, config, domain, ff):
    # Initialization
    archive = ca.create_archive(domain, config)
    fitness, features = ff.fitness_fun(init, domain)
    replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
    archive = update.update_map(replaced, replacement, archive, fitness, init, features)

    # Evolution
    for iGen in range(config['num_gens']):
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
        children = np.array([])
        while children.shape[0] < config['num_children']:
            new_children = cc.create_children(archive, domain, config)
            children = np.vstack([children, new_children]) if children.size else new_children

        fitness, features = ff.fitness_fun(children, domain)
        replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
        archive = update.update_map(replaced, replacement, archive, fitness, children, features)

    return archive

