import numpy as np
import random as rnd


class genome():
	def __init__(self, domain):
		self.domain = domain
		self.genes = np.random.normal(size=[self.domain['dof'], 1], scale=self.domain['init_weight_variance'])

	def mutate(self, probability=0.1, sigma=1.0):
		with np.nditer(self.genes, op_flags=['readwrite']) as it:
			for x in it:
				r = rnd.random()
				if r < probability:
					added = rnd.gauss(0,sigma)
					x[...] = x[...] + added

	def express(self):
		phenotype = self.genes
		return phenotype