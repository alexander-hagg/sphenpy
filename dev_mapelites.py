import sys
import yaml
sys.path.append('qd/mapelites')
from mapelites import *
from vis_archive import *
from domain.random import fitness_fun as ff

config = yaml.safe_load(open("qd/mapelites/config.yml"))
domain = yaml.safe_load(open("domain/random/domain.yml"))

archive = mapelites(config, domain, ff)
vis_archive(archive)


