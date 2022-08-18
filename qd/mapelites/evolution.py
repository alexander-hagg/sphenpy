from qd.mapelites.archive import mapelites_archive


def evolve(init, config, domain, ff):
    # Initialization
    archive = mapelites_archive(domain, config)
    fitness, features, phenotypes = ff(init)
    features = features[:,[domain['features'][0],domain['features'][1]]]
    improvement = []
    improvement.append(archive.update(fitness, features, init))

    # Evolution
    for iGen in range(config['num_gens']):
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
            children = archive.create_children()
            fitness, features, phenotypes = ff(children)
            features = features[:,[domain['features'][0],domain['features'][1]]]
            improvement.append(archive.update(fitness, features, children))

    return archive, improvement
