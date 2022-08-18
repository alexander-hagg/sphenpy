import numpy as np

def do(children, domain):
    print("Mutating")
    print(f'len(children): {len(children)}')
    # ranges = np.array(domain['par_ranges'])
    # print(f'ranges: {ranges}')
    # quit()
    # print(f'children: {children}')
    for child in children:
        print(child[0])
        child[0].mutate()
        # net['num_activation_genes'] = net['num_neurons'] * (net['num_layers'] + 1)
        # net['num_weight_genes'] = net['min_neurons']*net['min_neurons'] * (net['num_layers']+1)

        # mutation = np.random.randn(config['num_children'],domain['dof']) * config['mut_sigma']
        # mutation = mutation * (ranges[1]-ranges[0])
        # children = children + mutation
        # Limit ranges
        # toolow = children < ranges[0]
        # toohigh = children > ranges[1]
        # rangelowvalues = np.tile(ranges[0], (config['num_children'], 1))
        # rangehighvalues = np.tile(ranges[1], (config['num_children'], 1))
        # children[toolow] = rangelowvalues[toolow]
        # children[toohigh] = rangehighvalues[toohigh]
    return children
