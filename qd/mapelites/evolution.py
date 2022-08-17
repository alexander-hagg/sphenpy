import numpy as np
from qd.mapelites.archive import mapelites_archive
import qd.mapelites.niche_compete as compete
import qd.mapelites.update_archive as update
import qd.mapelites.create_children as cc

import matplotlib.pyplot as plt
from util import maptorange

def evolve(init, config, domain, ff):
    print(f'ff: {ff}')
    # Initialization
    arc = mapelites_archive(domain, config)
    fitness, features, phenotypes = ff(init)
    features = features[:,[domain['features'][0],domain['features'][1]]]
    replaced, replacement = compete.niche_compete(fitness, features, arc, domain, config)
    # archive = update.update_archive(replaced, replacement, archive, fitness, init, features)

    # Evolution
    # for iGen in range(config['num_gens']):
    #     if iGen%100 == 0:
    #         print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
    #     children = cc.create_children(archive, domain, config)

    #     fitness, features, phenotypes = ff(children)
    #     features = features[:,[domain['features'][0],domain['features'][1]]]
    #     # for i in range(features.shape[1]):
    #     #     features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][domain['features'][i]], domain['feat_ranges'][1][domain['features'][i]])

    #     replaced, replacement = compete.niche_compete(fitness, features, archive, domain, config)
    #     archive = update.update_archive(replaced, replacement, archive, fitness, children, features)

    # return archive
