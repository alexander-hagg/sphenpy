import numpy as np
import math
import yaml
from scipy.stats import qmc
# import imageio
from PIL import Image
import os
from domain.nsg_cppn import cppn


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/nsg_cppn/domain.yml"))

    domain['dof'] = domain['num_units'] * domain['dof_perblock']
    domain['par_ranges'] = np.tile(domain['par_ranges'], math.floor(domain['num_units']))
    img = Image.open('domain/nsg_cppn/' + domain['substrate_address'])
    img = img.resize((domain['grid_length'],domain['grid_length']))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img)
    domain['substrate'] = img.astype('bool')

    ranges = np.asarray(domain['par_ranges'])
    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(cppn.random())

    return domain, random_pop
