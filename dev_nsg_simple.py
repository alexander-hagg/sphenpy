import yaml
import time

from domain.nsg_simple import init, fitness_fun, express

domain, random_pop = init.do(1)

print("1")
phenotypes = express.do(random_pop, domain)
print("2")
plt = express.visualize(phenotypes[0], domain)
print("3")


# fitness, features = fitness_fun.get(random_pop, domain)
