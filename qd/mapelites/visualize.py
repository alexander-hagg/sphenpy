import matplotlib.pyplot as plt

def plot(archive, domain):
    plt.clf()
    plt.imshow(archive['fitness'], cmap='winter')
    plt.clim(0,1)
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    cbar = plt.colorbar()
    cbar.set_label(domain['labels'][-1])
    # plt.show()
    return plt
