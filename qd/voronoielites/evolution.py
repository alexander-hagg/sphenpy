from qd.voronoielites.archive import voronoi_archive
import statistics


def evolve(init, config, domain, ff):
    # Initialization
    archive = voronoi_archive(domain, config)
    fitness, features = ff(init)[0:2]
    improvement = []
    improvement.append(archive.update(fitness, features, init))

    # Evolution
    for iGen in range(config['num_gens']):
        children = archive.create_children()
        fitness, features = ff(children)[0:2]
        improvement.append(archive.update(fitness, features, children))
        if iGen%100 == 0:
            print('Generation: ' + str(iGen) + '/' + str(config['num_gens']))
            # if iGen > 99:
            #     print('Avg. improvement in last 100 gens: ' + str(statistics.mean(improvement[-99:])) + '%')

    return archive, improvement
