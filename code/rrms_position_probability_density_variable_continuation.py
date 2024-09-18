import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative 

from make_position_space_wavefunc import deuteron_wavefunc

def deut_density(Nmax, aHO, r):
    return deuteron_wavefunc(Nmax, aHO, r)[3]

r_model = np.linspace(0, 15, 500)

log_density = np.log(deuteron_wavefunc(20, 1.4, r_model)[3])

plt.plot(r_model, log_density)

def test_deut_wavefunc(r, Nmax, aHO):
        return deuteron_wavefunc(Nmax, aHO, r)[3]

d_log_density = np.gradient(deut_density(20, 1.4, r_model))

plt.plot(r_model, d_log_density)

plt.show()