import numpy as np


def random(num_neurons=2, num_layers=2):
    net = {}
    net['num_inputs'] = 2
    net['num_outputs'] = 1
    net['act_funcs'] = {0: gaussian, 1: tanh, 2: sigmoid, 3: unit}
    net['num_neurons'] = num_neurons
    net['num_layers'] = num_layers
    net['activations'] = np.random.randint(len(net['act_funcs']), size=[num_neurons, num_layers+1])
    net['weights'] = np.random.normal(size=[num_neurons*num_neurons, num_layers+1], scale=1.0)
    return net


def sample(binary_sample_grid, net, grid_length):
    output_grid = np.zeros([grid_length,grid_length], dtype=float)
    for x in range(binary_sample_grid.shape[0]):
        for y in range(binary_sample_grid.shape[1]):
            if binary_sample_grid[x,y]:
                # Scaled input
                input = 2 * np.array([x,y]) / grid_length - 1
                output_grid[x,y] = forward(input, net)
    return output_grid


def forward(input, net):
    activations = []
    activations.append(input)
    
    for layer in range(net['weights'].shape[1]):
        activation = np.zeros(net['num_neurons'])
        if layer < net['weights'].shape[1]-1:
            this_layer_num_neurons = net['num_neurons']
        else:
            this_layer_num_neurons = net['num_outputs']
        for hidden_neuron_id in range(this_layer_num_neurons):
            for input_neuron_id in range(net['num_neurons']):
                act_func_id = net['activations'][hidden_neuron_id, layer]
                activation[hidden_neuron_id] += net['weights'][hidden_neuron_id * net['num_neurons'] + input_neuron_id, layer] * net['act_funcs'][act_func_id](activations[layer][input_neuron_id])
        activations.append(activation)

    return activations[net['weights'].shape[1]][0]


def gaussian(x):
    return -2 * x * np.exp(-1 * x**2)


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def unit(x):
    return x
