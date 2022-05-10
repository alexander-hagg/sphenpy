import sys
import yaml

# Load QD algorithm
sys.path.append('qd/mapelites')
from mapelites import *
from vis_archive import *
config = yaml.safe_load(open("qd/mapelites/config.yml"))

# Load domain
from domain.random import fitness_fun as ff
domain = yaml.safe_load(open("domain/random/domain.yml"))

# Run QD and visualize
archive = mapelites(config, domain, ff)
vis_archive(archive)


