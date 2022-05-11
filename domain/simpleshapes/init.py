import numpy as np
import math
import yaml

def do(ninit_samples):
    domain = yaml.safe_load(open("domain/simpleshapes/domain.yml"))
    radius = 0.5
    t = np.linspace(0, 2*math.pi, num=int(domain['dof']/2), endpoint=False)
    x1 = radius*np.cos(t)
    y1 = radius*np.sin(t)
    domain['base'] = [x1,y1]
    random_pop = np.random.rand(ninit_samples,domain['dof'])
    random_pop = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    return domain, random_pop
