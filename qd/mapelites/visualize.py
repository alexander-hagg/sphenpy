import matplotlib.pyplot as plt

def plot(archive):
    plt.imshow(archive.get('fitness'), cmap='winter')
    plt.clim(0,1)
    plt.show()