import numpy as np
import random as rnd

def get_network(num_neurons=4, num_layers=2):
    assert num_layers > 0
    assert num_neurons > 0
    net = {}
    net['num_inputs'] = 2
    net['num_outputs'] = 1
    # net['act_funcs'] = {0: gaussian, 1: tanh, 2: sigmoid, 3: zero, 4: sin, 5: step}
    net['act_funcs'] = {0: gaussian, 1: tanh, 2: sigmoid, 3: sin, 4: cos}
    
    net['num_neurons'] = num_neurons
    net['num_layers'] = num_layers
    if num_neurons > net['num_inputs']:
        net['min_neurons'] = num_neurons
    else:
        net['min_neurons'] = net['num_inputs']
    return net

    
def get_genome_sizes(net):
    num_activation_genes = net['num_neurons'] * (net['num_layers'] + 1)
    num_weight_genes = net['min_neurons']*net['min_neurons'] * (net['num_layers']+1)
    return [num_activation_genes, num_weight_genes]


def get_genome(net):
    return np.concatenate((net['activations'].flatten(), net['weights'].flatten()))


def mutate(net, probability = 0.1, sigma=1.0):
    with np.nditer(net['activations'], op_flags=['readwrite']) as it:
        for x in it:
            if rnd.random() < probability:
                x[...] = rnd.randint(0,len(net['act_funcs'])-1)
    with np.nditer(net['weights'], op_flags=['readwrite']) as it:
        for x in it:
            if rnd.random() < probability:
                x[...] = x[...] + rnd.gauss(0,sigma)
    return net


def set_genome(genome, net):
    a = np.reshape(genome[0:net['num_neurons'] * (net['num_layers'] + 1)-1], [net['num_neurons'], net['num_layers'] + 1])
    b = np.reshape(genome[net['num_neurons'] * (net['num_layers'] + 1):end], [net['min_neurons'] * net['min_neurons'], net['num_layers'] + 1])
    # net['num_neurons'] * (net['num_layers'] + 1)
    # net['min_neurons'] * net['min_neurons'] * (net['num_layers']+1)
    # net['activations'] =     
    # net['weights'] = 


def random(num_neurons=3, num_layers=3, init_weight_variance=2.0):
    net = get_network(num_neurons, num_layers)
    net['activations'] = np.random.randint(len(net['act_funcs']), size=[net['num_neurons'], net['num_layers']+1])
    net['weights'] = np.random.normal(size=[net['min_neurons']*net['min_neurons'], net['num_layers']+1], scale=init_weight_variance)
    return net


def sample(binary_sample_grid, net):
    grid_length = binary_sample_grid.shape[0]
    output_grid = np.zeros([grid_length,grid_length], dtype=float)
    for x in range(binary_sample_grid.shape[0]):
        for y in range(binary_sample_grid.shape[1]):
            if binary_sample_grid[x,y]:
                # Scaled input
                input = 2 * np.array([x,y]) / grid_length - 1
                # input = 10 * input
                # print(input)
                output_grid[x,y] = forward(input, net)
    return output_grid


def forward(input, net):
    activations = np.zeros((net['weights'].shape[0],net['weights'].shape[1]+1))
    activations[0,0] = input[0]
    activations[1,0] = input[1]
    
    for layer in range(1,net['weights'].shape[1]+1):
        if layer < net['weights'].shape[1]:
            this_layer_num_neurons = net['num_neurons']
        else:
            this_layer_num_neurons = net['num_outputs']
        for neuron_id in range(this_layer_num_neurons):
            act = 0.0
            for input_neuron_id in range(net['min_neurons']):
                weight = net['weights'][neuron_id * net['min_neurons'] + input_neuron_id, layer-1]
                inp = activations[input_neuron_id,layer-1]
                act += weight * inp
            act_func_id = net['activations'][neuron_id, layer-1]
            act = net['act_funcs'][act_func_id](act)
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
