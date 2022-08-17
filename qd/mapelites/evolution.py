import numpy as np
import qd.mapelites.create_archive as ca
import qd.mapelites.niche_compete as compete
import qd.mapelites.update_archive as update
import qd.mapelites.create_children as cc

import matplotlib.pyplot as plt
from util import maptorange

def evolve(init, config, domain, ff):
    # Initialization
    archive = ca.create_archive(domain, config)
    fitness, features, phenotypes = ff(init)
    features = features[:,[domain['features'][0],domain['features'][1]]]
    # print(f'features: {features}')
    # for i in range(features.shape[1]):
    #     features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][domain['features'][i]], domain['feat_ranges'][1][domain['features'][i]])
    # print(f'features: {features}')
    # quit()
    replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
    archive = update.update_archive(replaced, replacement, archive, fitness, init, features)

    # Evolution
    for iGen in range(config['num_gens']):
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
        # children = np.array([])
        # while children.shape[0] < config['num_children']:
        #     new_children = cc.create_children(archive, domain, config)
        #     # print(f'new_children: {new_children}')
        #     children = np.vstack([children, new_children]) if children.size else new_children
        children = cc.create_children(archive, domain, config)

        fitness, features, phenotypes = ff(children)
        features = features[:,[domain['features'][0],domain['features'][1]]]
        # for i in range(features.shape[1]):
        #     features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][domain['features'][i]], domain['feat_ranges'][1][domain['features'][i]])

        replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
        # perc_improvement = 100.0*len(replaced)/fitness.shape[0]
        archive = update.update_archive(replaced, replacement, archive, fitness, children, features)

    return archive
