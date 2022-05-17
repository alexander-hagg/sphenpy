import numpy as np
import qd.voronoielites.create_archive as ca
import qd.voronoielites.update_archive as update
import qd.voronoielites.create_children as cc

def evolve(init, config, domain, ff):
    # Initialization
    archive = ca.create_archive(domain, config)
    fitness, features = ff.get(init, domain)
    archive = update.update_archive(archive, init, features, fitness, config, domain)
    
    # Evolution
    for iGen in range(config['num_gens']):
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
        children = np.array([])
        while children.shape[0] < config['num_children']:
            new_children = cc.create_children(archive, domain, config)
            children = np.vstack([children, new_children]) if children.size else new_children

        fitness, features = ff.get(children, domain)
        archive = update.update_archive(archive, children, features, fitness, config, domain)
    
    return archive

