import numpy as np
from scipy.spatial.distance import cdist


def update_archive(archive, genes, features, fitness, config, domain):
    all_genomes = np.vstack((archive['genes'],genes))
    all_features = np.vstack((archive['features'],features))
    all_fitness = np.vstack((archive['fitness'],fitness))
    while all_features.shape[0] > config['resolution']:
        distances = cdist(all_features, all_features)
        ind = np.unravel_index(np.argmin(distances[np.where(~np.eye(distances.shape[0],dtype=bool))], axis=None), distances.shape)
        lower = all_fitness[ind[0]] > all_fitness[ind[1]]
        all_genomes = np.delete(all_genomes, ind[int(lower)], 0)
        all_features = np.delete(all_features, ind[int(lower)], 0)
        all_fitness = np.delete(all_fitness, ind[int(lower)], 0)

    # Update archive
    archive.update({'genes': all_genomes})
    archive.update({'features': all_features})
    archive.update({'fitness': all_fitness})
    
    return archive
