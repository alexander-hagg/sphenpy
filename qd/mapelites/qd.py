import numpy as np
import qd.mapelites.create_archive as ca
import qd.mapelites.niche_compete as compete
import qd.mapelites.update_archive as update
import qd.mapelites.create_children as cc


def evolve(init, config, domain, ff):
    # Initialization
    archive = ca.create_archive(domain, config)
    fitness, features = ff(init)
    replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
    archive = update.update_archive(replaced, replacement, archive, fitness, init, features)

    # Evolution
    for iGen in range(config['num_gens']):
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
        children = np.array([])
        while children.shape[0] < config['num_children']:
            new_children = cc.create_children(archive, domain, config)
            children = np.vstack([children, new_children]) if children.size else new_children

        fitness, features = ff(children)
        replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
        # perc_improvement = 100.0*len(replaced)/fitness.shape[0]
        archive = update.update_archive(replaced, replacement, archive, fitness, children, features)

    return archive
