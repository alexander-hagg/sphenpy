import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot(archive, express, domain, config):
    plt.clf()
    if domain["plotscale"]:
        scale = 2 * config["resolution"]
    else:
        scale = 1
    for i in range(archive["genes"].shape[0]):
        for j in range(archive["genes"].shape[1]):
            genome = archive["genes"][i, j, :]
            phenotype = express.express_single(genome, domain)
            if phenotype is not None:
                dx = archive["features"][i, j, 0]
                dy = archive["features"][i, j, 1]
                fitness = archive["fitness"][i, j]
                if fitness > 1.0:
                    fitness = 1.0
                elif fitness < 0.0:
                    fitness = 0.0
                express.visualize_raw(
                    phenotype, [1 - fitness, fitness, 0], dx * scale, dy * scale
                )
    plt.axis("equal")
    plt.xlabel(domain["labels"][domain["features"][0]])
    plt.ylabel(domain["labels"][domain["features"][1]])
    # cbar = plt.colorbar()
    # norm = mpl.colors.Normalize(vmin=0,vmax=2)
    # N = 21
    # cmap = plt.get_cmap('jet',N)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ticks=np.linspace(0,2,N), boundaries=np.arange(-0.05,2.1,.1))
    # cbar.set_label(domain['labels'][-1])
    # plt.show()
    return plt
