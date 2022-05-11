import numpy as np

def create_children(archive, domain, config):
    # Randomly select parents and copy to children
    pool = archive['genes']
    pool = pool.reshape((pool.shape[0]*pool.shape[1], pool.shape[2]))
    selection = np.random.randint(0, pool.shape[0], config['num_children'])
    children = np.take(pool, selection, axis=0)

    # Mutate children
    ranges = np.array(domain['par_ranges'])
    mutation = np.random.randn(config['num_children'],domain['dof']) * config['mut_sigma']
    mutation = mutation * (ranges[1]-ranges[0])
    children = children + mutation

    children[children>1.0] = 1.
    children[children<0.0] = 0.

    return children
