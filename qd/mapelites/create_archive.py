import numpy as np


def create_archive(domain, config):
    edges = []
    res = []
    for i in range(len(domain['features'])):
        edges.append(np.linspace(0, 1, config['resolution']))
        res.append(config['resolution'])

    empty_archive = np.empty((res))
    empty_archive[:] = np.nan
    fitness = empty_archive
    genes = empty_archive
    genes = np.expand_dims(genes, 2)
    genes = np.tile(genes, (1, 1, domain['dof']))
    features = empty_archive
    features = np.expand_dims(features, 2)
    features = np.tile(features, (1, 1, len(domain['features'])))

    archive = {'edges': edges,
               'resolution': config['resolution'],
               'fitness': fitness,
               'features': features,
               'genes': genes,}
    return archive
