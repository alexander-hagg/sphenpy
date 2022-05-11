import numpy as np

def create_archive(domain, config):
    edges = []
    res = []
    for i in range(len(domain.get('features'))):
        edges.append(np.linspace(0, 1, config.get('resolution')))
        res.append(config.get('resolution'));
    
    empty_archive = np.empty((res))
    empty_archive[:] = np.nan
    fitness = empty_archive
    genes = empty_archive
    genes = np.expand_dims(genes, 2)
    genes = np.tile(genes, (1, 1, domain.get('dof')))
    features = np.tile(empty_archive, (domain.get('nfeatures'), 1, 1))
    
    archive = {'edges': edges,
               'resolution': config.get('resolution'),
               'fitness': fitness, 
               'features': features,
               'genes': genes,}
    return archive