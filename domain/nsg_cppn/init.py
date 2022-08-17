import numpy as np
import math
import yaml
from scipy.stats import qmc
# import imageio
from PIL import Image
import os
from domain.nsg_cppn import cppn, cust_mutation


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/nsg_cppn/domain.yml"))

    img = Image.open('domain/nsg_cppn/' + domain['substrate_address'])
    img = img.resize((domain['num_grid_cells'],domain['num_grid_cells']))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img)
    domain['substrate'] = img.astype('bool')

    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(cppn.cppn(domain['num_neurons'], domain['num_layers'], domain['init_weight_variance']))

    domain['dof'] = random_pop[0].get_genome().size

    # Do this last please
    domain['custom_mutation_fcn'] = lambda x: cust_mutation.do(x, domain)

    # from domain.nsg_cppn.cppn import cppn
    domain['gene_type'] = cppn.cppn

    return domain, random_pop
