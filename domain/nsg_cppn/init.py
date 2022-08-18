import yaml
from PIL import Image
import numpy as np
from domain.nsg_cppn.genome import cppn_genome


def do(ninit_samples):
    domain = yaml.safe_load(open("domain/nsg_cppn/domain.yml"))

    img = Image.open('domain/nsg_cppn/' + domain['substrate_address'])
    img = img.resize((domain['num_grid_cells'],domain['num_grid_cells']))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img)
    domain['substrate'] = img.astype('bool')

    random_pop = []
    for i in range(ninit_samples):
        random_pop.append(cppn_genome(domain))

    return domain, random_pop
