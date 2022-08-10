import numpy as np
import random as rnd


class cppn:
    def __init__(self, num_neurons=2, num_layers=2, sigma=2.0):
        assert num_layers > 0
        assert num_neurons > 0
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.num_inputs = 2
        self.num_outputs = 1
        self.act_funcs = {0:gaussian, 1:tanh, 2:sigmoid, 3:sin, 4:cos, 5:zero}
        if self.num_neurons > self.num_inputs:
            self.min_neurons = self.num_neurons
        else:
            self.min_neurons = self.num_inputs
        self.num_activation_genes = self.num_neurons * (self.num_layers + 1)
        self.num_weight_genes = self.min_neurons * self.min_neurons * (self.num_layers+1)
        self.activations = np.random.randint(len(self.act_funcs), size=[self.num_neurons, self.num_layers+1])
        self.weights = np.random.normal(size=[self.min_neurons*self.min_neurons, self.num_layers+1], scale=sigma)

    def get_genome(self):
        return np.concatenate((self.activations.flatten(), self.weights.flatten()))

    def mutate(self, probability=0.1, sigma=1.0):
        with np.nditer(self.activations, op_flags=['readwrite']) as it:
            for x in it:
                if rnd.random() < probability:
                    x[...] = rnd.randint(0,len(self.act_funcs)-1)
        with np.nditer(self.weights, op_flags=['readwrite']) as it:
            for x in it:
                if rnd.random() < probability:
                    x[...] = x[...] + rnd.gauss(0,sigma)

    def set_genome(self, genome):
        self.activations = np.reshape(genome[0:self.num_neurons * (self.num_layers + 1)], [self.num_neurons, self.num_layers + 1])
        self.weights = np.reshape(genome[self.num_neurons * (self.num_layers + 1):], [self.min_neurons * self.min_neurons, self.num_layers + 1])

    def sample(self, binary_sample_grid, domain):
        grid_length = binary_sample_grid.shape[0]
        output_grid = np.zeros([grid_length,grid_length], dtype=float)

        for x in range(binary_sample_grid.shape[0]):
            for y in range(binary_sample_grid.shape[1]):
                if binary_sample_grid[x,y]:
                    # Scaled input
                    input = 2 * np.array([x,y]) / grid_length - 1
                    output_grid[x,y] = self.forward(input)
        return output_grid

    def forward(self, input):
        activations = np.zeros((self.weights.shape[0],self.weights.shape[1]+1))
        activations[0,0] = input[0]
        activations[1,0] = input[1]

        for layer in range(1,self.weights.shape[1]+1):
            if layer < self.weights.shape[1]:
                this_layer_num_neurons = self.num_neurons
            else:
                this_layer_num_neurons = self.num_outputs
            for neuron_id in range(this_layer_num_neurons):
                act = 0.0
                for input_neuron_id in range(self.min_neurons):
                    weight = self.weights[neuron_id * self.min_neurons + input_neuron_id, layer-1]
                    inp = activations[input_neuron_id,layer-1]
                    act += weight * inp
                act_func_id = self.activations[neuron_id, layer-1]
                act = self.act_funcs[act_func_id](act)
                activations[neuron_id, layer] = act
        return activations[0,activations.shape[1]-1]


def gaussian(x):
    mu = 0
    sig = 1
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def zero(x):
    return 0 * x


def unit(x):
    return x/100


def step(x):
    if x >= 0:
        return np.ones(x.shape)
    else:
        return np.zeros(x.shape)


def bias(x):
    return np.ones(x.shape)
