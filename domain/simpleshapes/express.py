from get_natural_cubic_spline_model import *

def express(genomes, domain):
    model_6 = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=6)
    y_est_6 = model_6.predict(x)
    plt.plot(x, y_est_6, marker='.', label='n_knots = 6')
    plt.legend(); plt.show()