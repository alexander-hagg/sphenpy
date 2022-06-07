import numpy as np


def create_archive(domain, config):
    fitness = np.empty([0, 1])
    genes = np.empty([0, domain['dof']])
    features = np.empty([0, len(domain['features'])])

    archive = {'resolution': config['resolution'],
               'fitness': fitness,
               'features': features,
               'genes': genes,}
    return archive
