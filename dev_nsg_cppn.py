import yaml
import time

from domain.nsg_cppn import init, fitness_fun, express

domain, random_pop = init.do(1)

print("1")
phenotypes = express.do_surf(random_pop, domain)
print("2")
# plt = express.visualize_surf(phenotypes[0], domain)
print("3")
phenotypes = express.do(random_pop, domain)
print("4")
plt = express.visualize(phenotypes[0], domain)
print("5")

# fitness, features = fitness_fun.get(random_pop, domain)
