from qd.mapelites.archive import mapelites_archive
import statistics

def evolve(init, config, domain, ff):
    # Initialization
    archive = mapelites_archive(domain, config)
    fitness, features, phenotypes = ff(init)[0:3]
    improvement = []
    improvement.append(archive.update(fitness, features, init))

    # Evolution
    for iGen in range(config['num_gens']):
        children = archive.create_children()
        fitness, features, phenotypes = ff(children)[0:3]
        improvement.append(archive.update(fitness, features, children))
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
            if iGen > 99:
                print('Avg. improvement in last 100 gens: ' + str(statistics.mean(improvement[-99:])) + '%')

    return archive, improvement
