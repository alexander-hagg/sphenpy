import numpy as np
from templates.archive import archive
from templates.genome import genome


class mapelites_archive(archive):
    def __init__(self, domain, config):
        self.domain = domain
        self.config = config
        self.edges = []
        self.res = []
        for i in range(len(self.domain['features'])):
            self.edges.append(np.linspace(0, 1, self.config['resolution']))
            self.res.append(self.config['resolution'])

        self.fitness = np.full(self.res, np.nan)
        self.genes = np.full(self.res, genome)
        self.features = np.full(self.res, np.nan)
        self.features = np.expand_dims(self.features, 2)
        self.features = np.tile(self.features, (1, 1, len(domain['features'])))