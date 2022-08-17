import matplotlib.pyplot as plt


def plot(archive, domain, ucbplot=False):
    plt.clf()
    plt.imshow(archive['fitness'], cmap='winter')
    if not ucbplot:
        plt.clim(0, 1)
    plt.xlabel(domain['labels'][domain['features'][0]])
    plt.ylabel(domain['labels'][domain['features'][1]])
    cbar = plt.colorbar()
    if not ucbplot:
        cbar.set_label(domain['labels'][-1])
    else:
        cbar.set_label('Upper confidence bound')
    plt.show()
    return plt
