import numpy as np

def create_children(archive, domain, config):
    # Randomly select parents and copy to children
    pool = archive['genes']
    pool = pool.reshape((pool.shape[0]*pool.shape[1], pool.shape[2]))
    # Remove empty genomes
    invalid = np.isnan(pool).any(axis=1)
    pool = np.delete(pool, np.where(invalid), axis=0)
    selection = np.random.randint(0, pool.shape[0], config['num_children'])
    children = np.take(pool, selection, axis=0)

    # Mutate children
    ranges = np.array(domain['par_ranges'])
    mutation = np.random.randn(config['num_children'],domain['dof']) * config['mut_sigma']
    mutation = mutation * (ranges[1]-ranges[0])
    children = children + mutation
    # Check ranges
    # print('Checking ranges')
    # print(children>ranges[1])
    # print(children<ranges[0])

    # children[children>1.0] = 1.
    # children[children<0.0] = 0.

    return children
