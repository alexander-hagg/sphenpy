import numpy as np


def create_children(archive, domain, config):
    # Randomly select parents and copy to children
    pool = archive['genes']
    pool = pool.reshape((pool.shape[0]*pool.shape[1], pool.shape[2]))

    # Remove empty genomes
    pool = np.delete(pool, np.where(np.isnan(pool).any(axis=1)), axis=0)
    selection = np.random.randint(0, pool.shape[0], config['num_children'])
    children = np.take(pool, selection, axis=0)

    # Mutate children
    ranges = np.array(domain['par_ranges'])
    mutation = np.random.randn(config['num_children'],domain['dof']) * config['mut_sigma']
    mutation = mutation * (ranges[1]-ranges[0])
    children = children + mutation

    # Limit ranges
    toolow = children < ranges[0]
    toohigh = children > ranges[1]
    rangelowvalues = np.tile(ranges[0], (config['num_children'], 1))
    rangehighvalues = np.tile(ranges[1], (config['num_children'], 1))
    children[toolow] = rangelowvalues[toolow]
    children[toohigh] = rangehighvalues[toohigh]

    return children
